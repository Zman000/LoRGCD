from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass
class DatasetInfo:
    num_classes: int
    old_classes: np.ndarray
    new_classes: np.ndarray
    class_names: List[str]


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class GCDDataset(Dataset):
    def __init__(self, base: Dataset, labels: Sequence[int], labeled_mask: np.ndarray, transform):
        self.base = base
        self.labels = np.asarray(labels, dtype=np.int64)
        self.labeled_mask = np.asarray(labeled_mask, dtype=bool)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def _get_image(self, idx):
        item = self.base[idx]
        if isinstance(item, (tuple, list)):
            return item[0]
        return item

    def __getitem__(self, idx):
        img = self._get_image(idx)
        views = self.transform(img)
        y = int(self.labels[idx])
        is_labeled = bool(self.labeled_mask[idx])
        return views, y, is_labeled, idx


class CSVImageDataset(Dataset):
    """CSV with at least columns: path,label. Paths may be relative to root."""
    def __init__(self, csv_path: str, root: str = ""):
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not {"path", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV must contain columns 'path' and 'label'")
            for row in reader:
                p = row["path"]
                if root and not os.path.isabs(p):
                    p = os.path.join(root, p)
                self.samples.append((p, int(row["label"])))
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with Image.open(p) as im:
            img = im.convert("RGB")
        return img, y


def build_transforms(image_size: int = 224):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_t = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.3, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_t = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_t, eval_t


def _load_old_classes(
    num_classes: int,
    old_fraction: Optional[float],
    split_file: Optional[str],
) -> np.ndarray:
    if split_file:
        with open(split_file) as f:
            data = json.load(f)
        if isinstance(data, dict):
            vals = data.get("old_classes", data.get("known_classes", None))
            if vals is None:
                raise ValueError("split_file JSON dict must contain old_classes or known_classes")
        else:
            vals = data
        old = np.asarray(vals, dtype=np.int64)
    else:
        if old_fraction is None:
            raise ValueError("old_fraction must be resolved before _load_old_classes when no split_file is used")
        n_old = max(1, int(round(num_classes * old_fraction)))
        old = np.arange(n_old, dtype=np.int64)
    if np.any(old < 0) or np.any(old >= num_classes):
        raise ValueError("old class IDs outside [0, num_classes)")
    return np.unique(old)


def _make_labeled_mask(labels: np.ndarray, old_classes: np.ndarray, labeled_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in old_classes:
        ids = np.flatnonzero(labels == c)
        rng.shuffle(ids)
        n = max(1, int(round(labeled_fraction * len(ids)))) if len(ids) else 0
        mask[ids[:n]] = True
    return mask


def _get_base_dataset(name: str, root: str, train: bool, download: bool, csv_path: Optional[str]):
    lname = name.lower()
    if lname == "fake":
        # Dependency-light end-to-end smoke-test dataset. Never use for paper results.
        ds = datasets.FakeData(size=128 if train else 128, image_size=(3, 96, 96), num_classes=10,
                               transform=None, target_transform=None, random_offset=0)
        names = [f"class_{i}" for i in range(10)]
        labels = np.asarray([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    elif lname == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, download=download)
        names = list(ds.classes)
        labels = np.asarray(ds.targets)
    elif lname == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, download=download)
        names = list(ds.classes)
        labels = np.asarray(ds.targets)
    elif lname in {"imagefolder", "folder", "cub", "cars", "aircraft", "imagenet100", "herbarium19"}:
        path = os.path.join(root, "train") if train and os.path.isdir(os.path.join(root, "train")) else root
        ds = datasets.ImageFolder(path)
        names = list(ds.classes)
        labels = np.asarray(ds.targets)
    elif lname == "csv":
        if not csv_path:
            raise ValueError("--csv_path is required for dataset=csv")
        ds = CSVImageDataset(csv_path, root=root)
        labels = np.asarray(ds.targets)
        n = int(labels.max()) + 1
        names = [str(i) for i in range(n)]
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return ds, labels, names


def build_gcd_datasets(
    name: str,
    root: str,
    image_size: int = 224,
    old_fraction: Optional[float] = None,
    labeled_fraction: float = 0.5,
    split_file: Optional[str] = None,
    seed: int = 42,
    download: bool = False,
    csv_path: Optional[str] = None,
):
    train_base, labels, names = _get_base_dataset(name, root, True, download, csv_path)
    num_classes = len(np.unique(labels))
    lname = name.lower()
    if old_fraction is None:
        # Generic GCD protocol used by the original GCD/SimGCD benchmark families:
        # CIFAR-10: 5/10 old, CIFAR-100: 80/100 old, ImageNet-100: 50/100 old.
        # Fine-grained SSB datasets should use an explicit official split_file.
        old_fraction = 0.8 if lname == "cifar100" else 0.5
    if split_file is None and lname in {"cub", "cars", "aircraft", "herbarium19"}:
        import warnings
        warnings.warn(
            f"No split_file supplied for {name}. The deterministic first-class fallback is only "
            "for debugging; use the official SSB old-class split for paper comparisons.",
            RuntimeWarning,
        )
    # Requiring labels 0..K-1 keeps classifier/evaluation semantics explicit.
    uniq = np.unique(labels)
    if not np.array_equal(uniq, np.arange(num_classes)):
        raise ValueError("Dataset labels must be contiguous 0..K-1. Remap them before training.")
    old = _load_old_classes(num_classes, old_fraction, split_file)
    new = np.setdiff1d(np.arange(num_classes), old)
    labeled_mask = _make_labeled_mask(labels, old, labeled_fraction, seed)
    train_t, eval_t = build_transforms(image_size)
    train_ds = GCDDataset(train_base, labels, labeled_mask, TwoViewTransform(train_t))
    eval_ds = GCDDataset(train_base, labels, labeled_mask, eval_t)
    info = DatasetInfo(num_classes, old, new, names)
    return train_ds, eval_ds, info
