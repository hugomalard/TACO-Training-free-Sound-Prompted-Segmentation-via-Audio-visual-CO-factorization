"""Run one benchmark and summarize repeated random initializations."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from taco.data import load_table, make_loader, word_bank
from taco.factorization import conmf_bary, conmf_txt
from taco.metrics import (
    SemanticMeter,
    max_threshold_scores,
    minmax,
    per_class_detection_scores,
    resize_map,
)
from taco.models import audio_tokens, encode_bank, feats2shared, load_models, segment
from taco.paths import load_paths
from taco.similarity import get_closest


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    batch_size: int
    iterations: int
    temporal: bool
    drop_first_label: bool
    shuffle: bool
    image_tokens: str
    factorization: str
    seed: int = None
    share_clip_with_fcclip: bool = False
    dummy_fcclip: bool = False
    k: int = 8
    b1: float = 1.0
    b2: float = 1.0
    b3: float = 125.0
    b4: float = 0.0
    lr: float = 0.25
    positivity: bool = True
    semantic_classes: int = 0


SPECS = {
    "s4": BenchmarkSpec(
        name="s4",
        batch_size=1,
        iterations=1800,
        temporal=True,
        drop_first_label=False,
        shuffle=False,
        seed=None,
        image_tokens="fcclip",
        factorization="conmf_txt",
    ),
    "ms3": BenchmarkSpec(
        name="ms3",
        batch_size=4,
        iterations=1800,
        temporal=False,
        drop_first_label=True,
        shuffle=True,
        seed=42,
        image_tokens="fcclip",
        factorization="conmf_txt",
    ),
    "ade_sp": BenchmarkSpec(
        name="ade_sp",
        batch_size=20,
        iterations=1800,
        temporal=False,
        drop_first_label=False,
        shuffle=False,
        seed=None,
        image_tokens="fcclip",
        factorization="conmf_txt",
    ),
    "avss": BenchmarkSpec(
        name="avss",
        batch_size=18,
        iterations=1800,
        temporal=False,
        drop_first_label=False,
        shuffle=False,
        seed=None,
        image_tokens="openclip",
        factorization="conmf_txt",
        share_clip_with_fcclip=True,
        semantic_classes=71,
    ),
    "ade_sp_semantic": BenchmarkSpec(
        name="ade_sp_semantic",
        batch_size=12,
        iterations=1800,
        temporal=False,
        drop_first_label=False,
        shuffle=False,
        seed=None,
        image_tokens="fcclip",
        factorization="conmf_txt",
        dummy_fcclip=True,
    ),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _summarize(runs):
    keys = runs[0].keys()
    mean = {key: float(np.mean([run[key] for run in runs])) for key in keys}
    std = {key: float(np.std([run[key] for run in runs])) for key in keys}
    return mean, std


def _factorize(spec, image_tok, audio_tok, clap_text, clip_text, clap_model, barycenters):
    image_tok = image_tok.detach().clamp(min=0).cuda()
    audio_tok = audio_tok.detach().clamp(min=0).cuda()
    if spec.factorization == "conmf_bary":
        return conmf_bary(
            image_tok,
            audio_tok,
            spec.k,
            b1=spec.b1,
            b2=spec.b2,
            b3=spec.b3,
            b4=spec.b4,
            max_iter=spec.iterations,
            lr=spec.lr,
            barycenters=barycenters,
            positivity=spec.positivity,
            clap_model=clap_model,
        )
    return conmf_txt(
        image_tok,
        audio_tok,
        spec.k,
        b1=spec.b1,
        b2=spec.b2,
        b3=spec.b3,
        b4=spec.b4,
        max_iter=spec.iterations,
        lr=spec.lr,
        clap_text=clap_text,
        clip_text=clip_text,
        clap_model=clap_model,
        positivity=spec.positivity,
        temporal=spec.temporal,
    )


def _class_logits(factors, chosen, clip_text):
    selected = factors[torch.arange(factors.shape[0], device=factors.device), chosen]
    return get_closest(selected, clip_text.detach().to(selected.device)).T


def _paint(score, class_index, size=224):
    painted = score.float() * class_index.to(score.device)
    painted = torch.nn.functional.interpolate(
        painted.view(1, 1, *painted.shape[-2:]),
        size=(size, size),
        mode="nearest",
    )
    return painted.squeeze().round()


def infer_once(spec, loader, models, clip_text, clap_text, barycenters, n_classes, readout_clip=None):
    predictions, coarse, targets = [], [], []
    class_ids = []
    meter = SemanticMeter(n_classes) if spec.name in ("avss", "ade_sp_semantic") else None

    for index, batch in enumerate(loader):
        print(f"[{spec.name}] {index}", flush=True)
        images = batch["image"]
        audios = batch["audio"]
        masks = batch["mask"]
        tokens = audio_tokens(models.clap, audios)
        batched = [{"image": images[i]} for i in range(images.shape[0])]
        features = None
        if spec.image_tokens == "openclip":
            with torch.no_grad():
                dense = models.clip(batch["clip_image"].cuda(), typeF="base")
                image_tok = feats2shared(models.clip, dense)
        else:
            with torch.no_grad():
                features = models.fcclip.get_image_feats(batched)
                if spec.dummy_fcclip:
                    dummy = torch.randn(
                        (images.shape[0], spec.k, 768), device="cuda"
                    ).clamp(min=0)
                    models.fcclip(batched, text_emb=dummy, features=features, return_pooled=True)
                image_tok = feats2shared(models.clip, features["clip_vis_dense"])

        activation, factors, _, _, chosen, _ = _factorize(
            spec, image_tok, tokens, clap_text, clip_text, models.clap, barycenters
        )
        chosen = chosen.view(-1)
        maps = segment(
            models.fcclip,
            images,
            factors,
            None if spec.image_tokens == "openclip" else features,
        )
        if meter is None:
            side = int(activation.shape[1] ** 0.5)
            grid = activation.view(activation.shape[0], side, side, spec.k)
        else:
            logits = _class_logits(factors, chosen, readout_clip if readout_clip is not None else clip_text)

        for item in range(activation.shape[0]):
            factor = chosen[item]
            if meter is None:
                score = resize_map(minmax(maps[item][factor])).squeeze()
                coarse_score = resize_map(minmax(grid[item, :, :, factor])).squeeze()
                gt = masks[item].squeeze().float()
                predictions.append(score.cpu())
                coarse.append(coarse_score.cpu())
                targets.append(gt.cpu())
                if spec.name == "ade_sp":
                    class_ids.append(int(batch["class_id"][item]))
            elif spec.name == "avss":
                hard = (maps[item].argmax(dim=0) == factor).float()
                painted = _paint(hard, logits[item].argmax())
                meter.update(painted, masks[item].squeeze())
            else:
                soft = minmax(maps[item][factor]).round()
                painted = _paint(soft, logits[item].argmax() + 1)
                gt = masks[item].squeeze().float() * (int(batch["class_id"][item]) + 1)
                meter.update(painted, gt)

    if meter is not None:
        miou, fscore = meter.compute()
        return {"miou": miou, "fscore": fscore}
    if spec.name == "ade_sp":
        miou, average_precision = per_class_detection_scores(predictions, targets, class_ids)
        miou_u, average_precision_u = per_class_detection_scores(coarse, targets, class_ids)
        return {
            "miou": miou,
            "map": average_precision,
            "miou_u": miou_u,
            "map_u": average_precision_u,
        }
    miou, fscore = max_threshold_scores(predictions, targets)
    miou_u, fscore_u = max_threshold_scores(coarse, targets)
    return {"miou": miou, "fscore": fscore, "miou_u": miou_u, "fscore_u": fscore_u}


def _write_summary(path, summary):
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(destination)


def run_benchmark(name, runs=3, limit=None, paths=None, output=None):
    spec = SPECS[name]
    resolved = load_paths(paths)
    if spec.seed is not None:
        seed_all(spec.seed)
    frame = load_table(spec, resolved, limit=limit)
    labels = word_bank(spec, frame, resolved)
    print(f"{spec.name}: {len(frame)} samples, word bank {len(labels)}", flush=True)
    models = load_models(resolved, keep_visual_trunk=spec.image_tokens == "openclip")
    clip_text, clap_text, readout_clip = encode_bank(models, labels)
    if spec.share_clip_with_fcclip:
        models.fcclip.backbone.clip_model = models.clip.module.model
    barycenters = None
    if spec.factorization == "conmf_bary":
        audio = torch.load(resolved["bary_audio"], map_location="cpu").squeeze()
        image = torch.load(resolved["bary_image"], map_location="cpu").squeeze()
        barycenters = torch.cat((audio, image), 1)
    loader = make_loader(spec, frame, resolved)
    n_classes = spec.semantic_classes or len(labels)
    results = []
    for run in range(runs):
        print(f"{spec.name}: run {run + 1}/{runs}", flush=True)
        results.append(
            infer_once(
                spec, loader, models, clip_text, clap_text, barycenters, n_classes, readout_clip
            )
        )
        print(results[-1], flush=True)
        mean, std = _summarize(results)
        summary = {"benchmark": spec.name, "runs": results, "mean": mean, "std": std}
        _write_summary(output, summary)
        if output:
            print(f"wrote {output}", flush=True)
    return summary
