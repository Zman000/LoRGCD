from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_gcd_datasets
from metrics import evaluate_gcd
from models import GCDModel
from utils import save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--split_file", default=None)
    p.add_argument("--csv_path", default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("args", {})
    dataset = args.dataset or cfg.get("dataset", "cifar100")
    split_file = args.split_file or cfg.get("split_file")
    _, eval_ds, info = build_gcd_datasets(
        dataset,
        args.data_root,
        image_size=cfg.get("image_size", 224),
        old_fraction=cfg.get("old_fraction", None),
        labeled_fraction=cfg.get("labeled_fraction", 0.5),
        split_file=split_file,
        seed=cfg.get("seed", 42),
        csv_path=args.csv_path or cfg.get("csv_path"),
    )
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    num_classes = cfg.get("num_classes_override") or info.num_classes
    model = GCDModel(
        cfg.get("backbone", "vit_base_patch16_224.dino"),
        num_classes,
        cfg.get("proj_dim", 256),
        cfg.get("proj_hidden", 2048),
        cfg.get("proj_layers", 3),
        pretrained=False,
        freeze_blocks=0,
        classifier_scale=cfg.get("classifier_scale", 1.0),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ys, preds, zs, unlabeled = [], [], [], []
    for x, y, is_labeled, _ in tqdm(loader):
        out = model(x.to(device, non_blocking=True))
        ys.append(y.numpy())
        preds.append(out["logits"].argmax(-1).cpu().numpy())
        zs.append(out["semantic"].cpu().numpy())
        unlabeled.append((~is_labeled.bool()).numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    z = np.concatenate(zs)
    um = np.concatenate(unlabeled).astype(bool)
    metrics, mapping = evaluate_gcd(y[um], pred[um], info.old_classes)
    result = {**metrics.to_dict(), "checkpoint": args.checkpoint}
    print(json.dumps(result, indent=2))

    # `--output` may be either an output directory or an explicit .npz feature path.
    # This makes CLI behavior unambiguous for scripts while preserving the old directory mode.
    if args.output and str(args.output).lower().endswith(".npz"):
        feature_path = args.output
        out_dir = os.path.dirname(feature_path) or "."
    else:
        out_dir = args.output or os.path.dirname(args.checkpoint) or "."
        feature_path = os.path.join(out_dir, "features_eval.npz")
    os.makedirs(out_dir, exist_ok=True)
    save_json(result, os.path.join(out_dir, "evaluation.json"))
    save_json(mapping, os.path.join(out_dir, "evaluation_mapping.json"))
    np.savez_compressed(
        feature_path, z=z, y=y, pred=pred, unlabeled=um, old_classes=info.old_classes,
    )


if __name__ == "__main__":
    main()
