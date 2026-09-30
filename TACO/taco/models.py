"""Frozen CLAP, OpenCLIP, and FC-CLIP."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from taco.paths import ensure_repo_on_path

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_PROMPTS = "A photo of a {}"
AUDIO_PROMPTS = "This is a sound of {}"


class CLAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, typeF="base"):
        encoder = self.model.clap.audio_encoder
        inputs = inputs.to(encoder.base.htsat.bn0.weight.device)
        if typeF == "base":
            return encoder.base(inputs)
        if typeF == "projection":
            return encoder.projection(inputs)
        raise ValueError(typeF)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class CLIPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, typeF="base"):
        trunk = self.model.visual.trunk
        inputs = inputs.to(trunk.head.norm.weight.device)
        if typeF == "base":
            return trunk.forward_features(inputs)
        if typeF == "forward_head":
            return trunk.forward_head(inputs)
        if typeF == "head_only":
            return self.model.visual.head(inputs)
        raise ValueError(typeF)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def feats2shared(clip, image_features):
    batch = image_features.shape[0]
    spatial = image_features.shape[-2] * image_features.shape[-1]
    unpool = torch.swapaxes(image_features.flatten(-2, -1), 1, 2).reshape(batch * spatial, -1)
    vis_unpool = clip(unpool.unsqueeze(2).unsqueeze(3), typeF="forward_head")
    return clip(vis_unpool, typeF="head_only").view(batch, spatial, -1)


def _build_fcclip(paths):
    ensure_repo_on_path()
    import detectron2.utils.comm as comm
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.engine import default_setup
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.logger import setup_logger
    from fcclip import add_fcclip_config, add_maskformer2_config

    args = SimpleNamespace(
        config_file=paths["fcclip_config"],
        dist_url="tcp://127.0.0.1:50285",
        eval_only=True,
        machine_rank=0,
        num_gpus=1,
        num_machines=1,
        opts=["MODEL.WEIGHTS", paths["fcclip_weights"]],
        resume=False,
    )
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="fcclip")
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # Inference branches on this flag.
    model.training = False
    return model


def _patch_msclap_tokens():
    """Add the pre-pool HTS-AT sequence to the CLAP output as ``tokens`` when it is missing."""
    from msclap.models.htsat import HTSAT_Swin_Transformer

    if getattr(HTSAT_Swin_Transformer, "_taco_tokens", False):
        return
    original = HTSAT_Swin_Transformer.forward_features

    def forward_features(self, x):
        captured = {}

        def capture_first(_module, inputs):
            captured.setdefault("first", inputs[0])

        handle = self.avgpool.register_forward_pre_hook(capture_first)
        try:
            output = original(self, x)
        finally:
            handle.remove()
        if (
            isinstance(output, dict)
            and "tokens" not in output
            and getattr(self.config, "loss_type", None) != "clip_ce"
            and "first" in captured
        ):
            output = dict(output)
            output["tokens"] = captured["first"]
        return output

    HTSAT_Swin_Transformer.forward_features = forward_features
    HTSAT_Swin_Transformer._taco_tokens = True


def load_models(paths, keep_visual_trunk=False):
    import open_clip
    from msclap import CLAP

    _patch_msclap_tokens()
    clip, _ = open_clip.create_model_from_pretrained(paths["openclip"])
    clip = clip.cuda().eval()
    clap = CLAP(version="2023", use_cuda=False)
    clap.clap.eval()
    fcclip = _build_fcclip(paths)
    return SimpleNamespace(clip=clip, clap=clap, fcclip=fcclip, keep_visual_trunk=keep_visual_trunk)


def encode_bank(models, labels, readout_labels=None):
    import open_clip

    image_prompts = [IMAGE_PROMPTS.format(label) for label in labels]
    audio_prompts = [AUDIO_PROMPTS.format(label) for label in labels]
    tokens = open_clip.tokenizer.tokenize(image_prompts).cuda()
    clip_text = models.clip.encode_text(tokens).float()
    readout_clip = None
    if readout_labels:
        readout_tokens = open_clip.tokenizer.tokenize(
            [IMAGE_PROMPTS.format(label) for label in readout_labels]
        ).cuda()
        readout_clip = models.clip.encode_text(readout_tokens).float()
    models.clap.clap = models.clap.clap.cpu()
    clap_inputs = models.clap.preprocess_text(audio_prompts)
    clap_text = models.clap.clap.caption_encoder(clap_inputs)
    models.clip.transformer = None
    models.clap.clap.caption_encoder = None
    # msclap.CLAP is not an nn.Module, so DataParallel.cuda() does not move it.
    models.clap.clap.cuda()
    models.clap = torch.nn.DataParallel(CLAPWrapper(models.clap)).eval()
    if not models.keep_visual_trunk:
        models.clip.visual.trunk.stages = None
    models.clip = torch.nn.DataParallel(CLIPWrapper(models.clip)).cuda().eval()
    return clip_text, clap_text, readout_clip


def audio_tokens(clap_model, audio_paths):
    preprocessed = clap_model.module.model.preprocess_audio(list(audio_paths), True)
    preprocessed = preprocessed.reshape(preprocessed.shape[0], preprocessed.shape[2])
    with torch.no_grad():
        encoded = clap_model(preprocessed)
    return torch.swapaxes(encoded["tokens"], 1, 2)


def prepare_fcclip(fcclip, k):
    fcclip.test_num_templates = [1 for _ in range(k)]
    fcclip.category_overlapping_mask = torch.zeros(k, device="cuda")


def segment(fcclip, images, factors, features=None):
    prepare_fcclip(fcclip, factors.shape[1])
    batched = [{"image": images[i]} for i in range(images.shape[0])]
    with torch.no_grad():
        if features is None:
            outputs = fcclip(batched, text_emb=factors)
        else:
            outputs = fcclip(batched, text_emb=factors, features=features)
    return [output["sem_seg"] for output in outputs]
