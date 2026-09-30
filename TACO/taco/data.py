"""Dataset readers."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

IMAGE_SIZE = 764
MASK_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# The CSVs still store the old capitalized mount names.
_STORED_ROOTS = (
    ("/tsi/dcase/AVSBench-semantic", "/tsi/dcase/avsbench_semantic"),
    ("/tsi/dcase/AVSBench", "/tsi/dcase/avsbench"),
)


def resolve_dataset_path(path):
    text = str(path)
    if os.path.exists(text):
        return text
    for old, new in _STORED_ROOTS:
        if text == old or text.startswith(old + "/"):
            return new + text[len(old) :]
    return text

CLIP_PREPROCESS = Compose(
    [
        Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


def rewrite_frame_name(path):
    """Multi-source frames are stored as ``{stem}.mp4_{index}.ext``."""
    parts = str(path).split("/")
    name, ext = parts[-1].rsplit("_", 1)
    parts[-1] = f"{name}.mp4_{ext}"
    return "/".join(parts)


def rgb2cls(seg):
    """ADE20K palette: class id = (R / 10) * 256 + G."""
    if torch.is_tensor(seg):
        array = seg.detach().cpu().numpy()
    else:
        array = np.asarray(seg)
    if array.shape[0] in (1, 3) and array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    red = array[:, :, 0]
    green = array[:, :, 1]
    return torch.tensor((red / 10).astype(np.int32) * 256 + green.astype(np.int32))


def _fcclip_image(path):
    image = Image.open(path).convert("RGB")
    tensor = torchvision.transforms.functional.pil_to_tensor(image).float()
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


def _clip_image(path):
    return CLIP_PREPROCESS(Image.open(path).convert("RGB"))


def _binary_mask(path):
    mask = torchvision.transforms.functional.pil_to_tensor(Image.open(path))
    if mask.shape[0] > 1:
        mask = mask[:1]
    return (mask.float() / 255).int()


def _sorted_files(directory, extensions):
    directory = Path(directory)
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    return sorted(files, key=lambda path: path.name)


def _ade_binary_mask(path, class_id):
    decoded = rgb2cls(torchvision.transforms.functional.pil_to_tensor(Image.open(path).convert("RGB")))
    resized = torch.nn.functional.interpolate(
        decoded.float().view(1, 1, *decoded.shape),
        size=(MASK_SIZE, MASK_SIZE),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)
    return (resized == class_id).int()


def v2_palette(num_cls=71):
    palette = np.zeros((num_cls, 3), dtype=np.int32)
    for label in range(num_cls):
        value = label
        bit = 0
        while value > 0:
            palette[label, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[label, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[label, 2] |= ((value >> 2) & 1) << (7 - bit)
            bit += 1
            value >>= 3
    return palette


def color_mask_to_label(path, palette):
    mask = np.array(Image.open(path).convert("RGB")).astype("int32")
    semantic = [np.all(mask == colour, axis=-1) for colour in palette]
    semantic = np.stack(semantic, axis=-1).astype(np.float32)
    label = torch.from_numpy(np.argmax(semantic, axis=-1)).float()
    return torch.nn.functional.interpolate(
        label.view(1, 1, *label.shape),
        size=(MASK_SIZE, MASK_SIZE),
        mode="nearest",
    ).squeeze(0)


def audioset_labels(metadata_csv, drop_first):
    labels = pd.read_csv(metadata_csv)["label"].unique()
    if drop_first:
        labels = labels[1:]
    return [str(label) for label in labels]


def _limit(frame, limit):
    if limit is None:
        return frame
    return frame.iloc[:limit].reset_index(drop=True)


def load_table(spec, paths, limit=None):
    if spec.name in ("s4", "ms3"):
        frame = pd.read_csv(paths["avsbench_metadata"])
        source = "single" if spec.name == "s4" else "multi"
        frame = frame[frame["source_type"] == source]
        frame = frame[frame["split"] == "test"]
        if spec.name == "s4":
            frame = frame.drop_duplicates(subset="video_id")
        if spec.shuffle:
            frame = frame.sample(frac=1).reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
        if spec.name == "ms3":
            frame["frame"] = [rewrite_frame_name(value) for value in frame["frame"]]
        return _limit(frame, limit)

    if spec.name in ("ade_sp", "ade_sp_semantic"):
        frame = pd.read_csv(paths["ade_metadata"]).reset_index(drop=True)
        return _limit(frame, limit)

    if spec.name == "avss":
        frame = pd.read_csv(paths["avss_metadata"])
        frame = frame[frame["source_type"] == "v2"].reset_index(drop=True)
        return _limit(frame, limit)
    raise KeyError(spec.name)


def word_bank(spec, frame, paths):
    if spec.name == "avss":
        with open(paths["avss_label2idx"], "r") as handle:
            return list(json.load(handle).keys())
    if spec.name == "ade_sp_semantic":
        return frame["ade_class"].unique().tolist()
    return audioset_labels(paths["audioset_metadata"], spec.drop_first_label)


class _PathDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


class AVSBenchFrames(_PathDataset):
    def __init__(self, frame, root):
        root = Path(root)
        records = [
            {
                "image": root / row.frame,
                "audio": resolve_dataset_path(row.file_path),
                "mask": root / row.mask,
                "class_id": -1,
            }
            for row in frame.itertuples(index=False)
        ]
        super().__init__(records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "image": _fcclip_image(row["image"]),
            "audio": row["audio"],
            "mask": _binary_mask(row["mask"]),
            "class_id": row["class_id"],
        }


class AVSBenchVideos(_PathDataset):
    def __init__(self, frame, root):
        root = Path(root)
        records = []
        for row in frame.itertuples(index=False):
            image_dir = root / Path(row.frame).parent
            audio_dir = Path(resolve_dataset_path(row.file_path)).parent
            mask_dir = root / Path(row.mask).parent
            images = _sorted_files(image_dir, IMAGE_EXTS)
            audios = _sorted_files(audio_dir, AUDIO_EXTS)
            masks = _sorted_files(mask_dir, IMAGE_EXTS)
            count = min(len(images), len(audios), len(masks))
            if count == 0:
                raise FileNotFoundError(f"No frames for video {row.video_id} in {image_dir}")
            records.append(
                {
                    "image": images[:count],
                    "audio": [str(path) for path in audios[:count]],
                    "mask": masks[:count],
                    "class_id": -1,
                }
            )
        super().__init__(records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "image": torch.stack([_fcclip_image(path) for path in row["image"]]),
            "audio": row["audio"],
            "mask": torch.stack([_binary_mask(path) for path in row["mask"]]),
            "class_id": row["class_id"],
        }


class ADEBinary(_PathDataset):
    def __init__(self, frame, paths):
        records = []
        for row in frame.itertuples(index=False):
            stem = str(row.image_location).rsplit(".", 1)[0]
            records.append(
                {
                    "image": Path(paths["ade_frames"]) / row.image_location,
                    "audio": str(Path(paths["vggsound_root"]) / row.file_path),
                    "mask": Path(paths["ade_annotations"]) / f"{stem}_seg.png",
                    "class_id": int(row.ade_class_id),
                    "ade_class_id": int(row.ade_class_id),
                }
            )
        super().__init__(records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "image": _fcclip_image(row["image"]),
            "audio": row["audio"],
            "mask": _ade_binary_mask(row["mask"], row["ade_class_id"]),
            "class_id": row["class_id"],
        }


class ADESemantic(_PathDataset):
    def __init__(self, frame, paths):
        classes = frame["ade_class"].unique().tolist()
        records = []
        for row in frame.itertuples(index=False):
            stem = str(row.image_location).rsplit(".", 1)[0]
            records.append(
                {
                    "image": Path(paths["ade_frames"]) / row.image_location,
                    "audio": str(Path(paths["vggsound_root"]) / row.file_path),
                    "mask": Path(paths["ade_annotations"]) / f"{stem}_seg.png",
                    "class_id": classes.index(row.ade_class),
                    "ade_class_id": int(row.ade_class_id),
                }
            )
        super().__init__(records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "image": _fcclip_image(row["image"]),
            "audio": row["audio"],
            "mask": _ade_binary_mask(row["mask"], row["ade_class_id"]),
            "class_id": row["class_id"],
        }


class AVSSFrames(_PathDataset):
    def __init__(self, frame, paths):
        self.palette = v2_palette()
        root = Path(paths["avss_root"])
        with open(paths["avss_label2idx"], "r") as handle:
            index = json.load(handle)
        records = []
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "image": root / row.frame,
                    "audio": resolve_dataset_path(row.file_path),
                    "mask": str(root / row.mask).replace("labels_semantic", "labels_rgb"),
                    "class_id": int(index.get(str(row.label).split("_")[0], -1)),
                }
            )
        super().__init__(records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "image": _fcclip_image(row["image"]),
            "clip_image": _clip_image(row["image"]),
            "audio": row["audio"],
            "mask": color_mask_to_label(row["mask"], self.palette),
            "class_id": row["class_id"],
        }


def _collate(batch):
    collated = {
        "image": torch.stack([item["image"] for item in batch]),
        "audio": [item["audio"] for item in batch],
        "mask": torch.stack([item["mask"] for item in batch]),
        "class_id": torch.tensor([item["class_id"] for item in batch]),
    }
    if "clip_image" in batch[0]:
        collated["clip_image"] = torch.stack([item["clip_image"] for item in batch])
    return collated


def _collate_video(batch):
    if len(batch) != 1:
        raise ValueError("Temporal S4 evaluation uses one video per batch.")
    sample = batch[0]
    sample["class_id"] = torch.tensor([sample["class_id"]])
    return sample


def make_loader(spec, frame, paths):
    if spec.name == "ms3":
        dataset = AVSBenchFrames(frame, paths["avsbench_root"])
        collate = _collate
    elif spec.name == "s4":
        dataset = AVSBenchVideos(frame, paths["avsbench_root"])
        collate = _collate_video
    elif spec.name == "ade_sp":
        dataset = ADEBinary(frame, paths)
        collate = _collate
    elif spec.name == "ade_sp_semantic":
        dataset = ADESemantic(frame, paths)
        collate = _collate
    elif spec.name == "avss":
        dataset = AVSSFrames(frame, paths)
        collate = _collate
    else:
        raise KeyError(spec.name)
    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
