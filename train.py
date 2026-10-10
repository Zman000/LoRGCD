from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets import build_gcd_datasets
from geometry import geometry_aware_mean_shift, global_mahalanobis_affinity, local_riemannian_affinity
from losses import (
    entropy_of_mean,
    instance_nt_xent,
    riemannian_affinity_distillation,
    soft_cross_entropy,
    supervised_contrastive,
    symmetric_distillation,
)
from metrics import evaluate_gcd
from models import GCDModel
from utils import ensure_dir, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser("LoRGCD training")
    # Data
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--data_root", required=True)
    p.add_argument("--csv_path", default=None)
    p.add_argument("--split_file", default=None, help="JSON list/dict of official old/known class IDs")
    p.add_argument("--old_fraction", type=float, default=None,
                   help="Override old-class fraction. Defaults: CIFAR-100=0.8, otherwise 0.5; use split_file for SSB")
    p.add_argument("--labeled_fraction", type=float, default=0.5)
    p.add_argument("--download", action="store_true")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    # Model
    p.add_argument("--backbone", default="vit_base_patch16_224.dino")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--freeze_blocks", type=int, default=11,
                   help="For ViT-B/16, 11 freezes all but the final transformer block")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_layers", type=int, default=3)
    p.add_argument("--classifier_scale", type=float, default=1.0)
    p.add_argument("--num_classes_override", type=int, default=None,
                   help="Use for class-count mismatch/unknown-K robustness experiments")
    # Optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--disable_balanced_sampler", action="store_true",
                   help="Official SimGCD balances labeled and unlabeled examples")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    # Baseline losses
    p.add_argument("--student_temp", type=float, default=0.10)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp", type=float, default=0.07)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)
    p.add_argument("--sup_weight", type=float, default=0.35,
                   help="Official SimGCD supervised-vs-unsupervised mixture")
    p.add_argument("--memax_weight", type=float, default=2.0)
    p.add_argument("--sup_temp", type=float, default=0.10)
    p.add_argument("--rep_temp", type=float, default=1.0,
                   help="Official SimGCD InfoNCE temperature default")
    p.add_argument("--supcon_temp", type=float, default=0.07)
    # LoRGCD geometry
    p.add_argument("--geo_k", type=int, default=16)
    p.add_argument("--geo_rank", type=int, default=8)
    p.add_argument("--geo_tau", type=float, default=1.0)
    p.add_argument("--geo_shrinkage", type=float, default=0.10)
    p.add_argument("--geo_steps", type=int, default=1)
    p.add_argument("--lambda_rad", type=float, default=0.50,
                   help="Riemannian Affinity Distillation weight")
    p.add_argument("--lambda_gmd", type=float, default=0.50,
                   help="Geometry-aware Mean-Shift Distillation weight")
    p.add_argument("--anchor_strength", type=float, default=1.0)
    p.add_argument("--geometry_mode", choices=["local", "euclidean", "global"], default="local",
                   help="local=LoRGCD, euclidean=isotropic kNN control, global=one global Mahalanobis metric")
    p.add_argument("--disable_local_metric", action="store_true",
                   help="Legacy alias for --geometry_mode euclidean")
    p.add_argument("--disable_supervised_anchors", action="store_true")
    p.add_argument("--disable_tangent_alignment", action="store_true",
                   help="Ablation: use local standardized displacement without tangent-bundle consistency")
    p.add_argument("--fixed_tangent_rank", action="store_true",
                   help="Ablation: force every neighborhood to use geo_rank instead of participation-ratio adaptive rank")
    # Logging
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/lorgcd")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def build_loaders(args):
    train_ds, eval_ds, info = build_gcd_datasets(
        args.dataset,
        args.data_root,
        image_size=args.image_size,
        old_fraction=args.old_fraction,
        labeled_fraction=args.labeled_fraction,
        split_file=args.split_file,
        seed=args.seed,
        download=args.download,
        csv_path=args.csv_path,
    )
    sampler = None
    shuffle = True
    if not args.disable_balanced_sampler:
        lm = np.asarray(train_ds.labeled_mask, dtype=bool)
        n_l, n_u = int(lm.sum()), int((~lm).sum())
        if n_l > 0 and n_u > 0:
            # Same principle as official SimGCD: total probability mass of labeled and
            # unlabeled subsets is balanced without using novel labels.
            weights = np.where(lm, 1.0, n_l / float(n_u))
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            num_samples=len(weights), replacement=True)
            shuffle = False
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    return train_loader, eval_loader, info


def set_lr(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        fac = (epoch + 1) / max(1, args.warmup_epochs)
    else:
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        fac = 0.5 * (1.0 + np.cos(np.pi * t))
    lr = args.lr * fac
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def teacher_temperature(epoch: int, args) -> float:
    n = max(0, int(args.warmup_teacher_temp_epochs))
    if n > 0 and epoch < n:
        if n == 1:
            return float(args.teacher_temp)
        t = epoch / float(n - 1)
        return float((1.0 - t) * args.warmup_teacher_temp + t * args.teacher_temp)
    return float(args.teacher_temp)


def compute_loss(model, batch, device, args, epoch: int = 0):
    (x1, x2), y, is_labeled, _ = batch
    x1 = x1.to(device, non_blocking=True)
    x2 = x2.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    lm = is_labeled.to(device, non_blocking=True).bool()

    out1 = model(x1)
    out2 = model(x2)

    # Baseline: instance SSL + supervised contrastive + parametric self-distillation.
    l_inst = instance_nt_xent(out1["z"], out2["z"], args.rep_temp)
    l_supcon = supervised_contrastive(out1["z"], out2["z"], y, lm, args.supcon_temp)
    t_temp = teacher_temperature(epoch, args)
    l_unsup = symmetric_distillation(
        out1["logits"], out2["logits"], args.student_temp, t_temp
    )
    if lm.any():
        l_sup = 0.5 * (
            F.cross_entropy(out1["logits"][lm] / args.sup_temp, y[lm])
            + F.cross_entropy(out2["logits"][lm] / args.sup_temp, y[lm])
        )
    else:
        l_sup = out1["z"].new_tensor(0.0)
    # SimGCD-style batch marginal entropy maximization counters old-class/imbalance bias.
    h_mean = entropy_of_mean(torch.cat([out1["logits"], out2["logits"]], dim=0) / args.student_temp)

    # Geometry teacher on the average of two views for stability.
    z_teacher = F.normalize(
        0.5 * (out1["semantic"].detach() + out2["semantic"].detach()), dim=-1
    )
    geo_mode = "euclidean" if args.disable_local_metric else args.geometry_mode
    common_geo = dict(
        z=z_teacher, k=args.geo_k, tau_geo=args.geo_tau,
        labels=None if args.disable_supervised_anchors else y,
        labeled_mask=None if args.disable_supervised_anchors else lm,
        supervised_anchor_strength=args.anchor_strength,
    )
    if geo_mode == "global":
        aff, geo_stats = global_mahalanobis_affinity(
            **common_geo, covariance_shrinkage=args.geo_shrinkage
        )
    else:
        aff, geo_stats = local_riemannian_affinity(
            **common_geo, rank=args.geo_rank, shrinkage=args.geo_shrinkage,
            use_local_metric=(geo_mode == "local"),
            use_tangent_alignment=(geo_mode == "local" and not args.disable_tangent_alignment),
            adaptive_rank=(not args.fixed_tangent_rank),
        )

    # RAD: make the trainable cosine embedding reproduce the anisotropic manifold graph.
    l_rad = 0.5 * (
        riemannian_affinity_distillation(out1["semantic"], aff, args.rep_temp)
        + riemannian_affinity_distillation(out2["semantic"], aff, args.rep_temp)
    )

    # GMD: manifold-smoothed views generate more stable teacher assignments.
    z1_ms = geometry_aware_mean_shift(out1["semantic"].detach(), aff, args.geo_steps)
    z2_ms = geometry_aware_mean_shift(out2["semantic"].detach(), aff, args.geo_steps)
    with torch.no_grad():
        q1 = F.softmax(model.classifier(z1_ms) / t_temp, dim=-1)
        q2 = F.softmax(model.classifier(z2_ms) / t_temp, dim=-1)
    l_gmd = 0.5 * (
        soft_cross_entropy(out1["logits"] / args.student_temp, q2)
        + soft_cross_entropy(out2["logits"] / args.student_temp, q1)
    )

    # Exact loss *weighting pattern* of SimGCD: unsupervised cluster/representation
    # terms receive (1-sup_weight), supervised terms receive sup_weight. Entropy
    # maximization is inside the cluster branch; log(K) is an irrelevant constant.
    cluster_branch = l_unsup - args.memax_weight * h_mean
    rep_branch = l_inst
    baseline = (
        (1.0 - args.sup_weight) * cluster_branch
        + args.sup_weight * l_sup
        + (1.0 - args.sup_weight) * rep_branch
        + args.sup_weight * l_supcon
    )
    total = baseline + args.lambda_rad * l_rad + args.lambda_gmd * l_gmd
    logs = {
        "loss": total.detach(), "inst": l_inst.detach(), "supcon": l_supcon.detach(),
        "unsup": l_unsup.detach(), "sup": l_sup.detach(), "entropy": h_mean.detach(),
        "rad": l_rad.detach(), "gmd": l_gmd.detach(),
        "geo_cond": geo_stats.mean_condition, "geo_anis": geo_stats.mean_anisotropy,
        "geo_nncos": geo_stats.mean_neighbor_cosine,
        "geo_tangent_mismatch": geo_stats.mean_tangent_mismatch,
        "geo_rank_mean": geo_stats.mean_tangent_rank,
    }
    return total, logs


@torch.no_grad()
def evaluate(model, loader, info, device):
    model.eval()
    ys, preds, z_all, unlabeled = [], [], [], []
    for views, y, is_labeled, _ in tqdm(loader, desc="eval", leave=False):
        x = views.to(device, non_blocking=True)
        out = model(x)
        ys.append(y.numpy())
        preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
        z_all.append(out["semantic"].cpu().numpy())
        unlabeled.append((~is_labeled.bool()).numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    z = np.concatenate(z_all)
    um = np.concatenate(unlabeled).astype(bool)
    metrics, mapping = evaluate_gcd(y[um], pred[um], info.old_classes)
    return metrics.to_dict(), mapping, z, y, um


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output)
    save_json(vars(args), os.path.join(args.output, "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, eval_loader, info = build_loaders(args)
    num_classes = args.num_classes_override or info.num_classes
    save_json({
        "num_classes": info.num_classes,
        "old_classes": info.old_classes,
        "new_classes": info.new_classes,
        "class_names": info.class_names,
        "classifier_classes": num_classes,
    }, os.path.join(args.output, "dataset_info.json"))

    model = GCDModel(
        args.backbone, num_classes, args.proj_dim, args.proj_hidden, args.proj_layers,
        pretrained=not args.no_pretrained, freeze_blocks=args.freeze_blocks,
        classifier_scale=args.classifier_scale,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        # Do not weight-decay biases/norm/1D parameters, following DINO/SimGCD practice.
        reg, no_reg = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_reg if (name.endswith(".bias") or p.ndim == 1) else reg).append(p)
        groups = [{"params": reg}, {"params": no_reg, "weight_decay": 0.0}]
        optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    history = []
    # CVPR-safe protocol: checkpoint selection must not use hidden evaluation labels.
    # We log evaluation metrics when requested, but primary paper results should use last.pt
    # (or a separately defined validation protocol), never a label-selected "best" epoch.
    for epoch in range(args.epochs):
        model.train()
        lr = set_lr(optimizer, epoch, args)
        meter = defaultdict(float)
        n_batches = 0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, logs = compute_loss(model, batch, device, args, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n_batches += 1
            for k, v in logs.items():
                meter[k] += float(v.item() if torch.is_tensor(v) else v)
            pbar.set_postfix(loss=f"{meter['loss']/n_batches:.3f}", rad=f"{meter['rad']/n_batches:.3f}")

        row = {k: v / max(1, n_batches) for k, v in meter.items()}
        row.update({"epoch": epoch + 1, "lr": lr, "seconds": time.time() - start})

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics, mapping, _, _, _ = evaluate(model, eval_loader, info, device)
            row.update({f"eval_{k}": v for k, v in metrics.items()})
            # Mapping is saved only as an evaluation diagnostic; it is not used for model selection.
            save_json(mapping, os.path.join(args.output, f"mapping_epoch_{epoch+1}.json"))
        history.append(row)
        save_json(history, os.path.join(args.output, "history.json"))

        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
                       os.path.join(args.output, f"epoch_{epoch+1}.pt"))
        print(json.dumps(row, indent=2))

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               os.path.join(args.output, "last.pt"))


if __name__ == "__main__":
    main()
