from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from geometry import local_graph_diagnostics, local_riemannian_affinity


def make_data(classes=4, per_class=256, dim=32, tangent_dim=6, tangent_std=2.0,
              normal_std=0.10, mean_gap=0.45, seed=0):
    rng = np.random.default_rng(seed)
    # Shared tangent subspace in first tangent_dim coordinates; class separation is placed
    # in locally low-variance normal coordinates to instantiate Proposition 1.
    xs, ys = [], []
    for c in range(classes):
        x = np.zeros((per_class, dim), dtype=np.float32)
        x[:, :tangent_dim] = tangent_std * rng.normal(size=(per_class, tangent_dim))
        x[:, tangent_dim:] = normal_std * rng.normal(size=(per_class, dim - tangent_dim))
        axis = tangent_dim + (c % max(1, dim - tangent_dim))
        x[:, axis] += mean_gap
        xs.append(x)
        ys.append(np.full(per_class, c, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # Random orthogonal mixing prevents axis-aligned shortcuts.
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    x = x @ q.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return torch.tensor(x), torch.tensor(y)


def main():
    p = argparse.ArgumentParser("Synthetic validation of local anisotropic geometry")
    p.add_argument("--ratios", default="1,2,4,8,16")
    p.add_argument("--normal_std", type=float, default=0.10)
    p.add_argument("--mean_gap", type=float, default=0.45)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--output", default="synthetic_anisotropy.csv")
    args = p.parse_args()

    rows = []
    for ratio in [float(v) for v in args.ratios.split(",")]:
        # variance ratio is ratio; standard deviation ratio is sqrt(ratio)
        tangent_std = args.normal_std * np.sqrt(ratio)
        z, y = make_data(tangent_std=tangent_std, normal_std=args.normal_std,
                         mean_gap=args.mean_gap, seed=123)
        a_e, _ = local_riemannian_affinity(z, k=args.k, rank=args.rank, use_local_metric=False)
        # Disable tangent alignment here to isolate Proposition 1 (inverse-covariance effect).
        a_l, stats = local_riemannian_affinity(
            z, k=args.k, rank=args.rank, use_local_metric=True, use_tangent_alignment=False
        )
        de = local_graph_diagnostics(a_e, y)
        dl = local_graph_diagnostics(a_l, y)
        row = {
            "variance_ratio": ratio,
            "euclidean_purity": de["neighbor_purity"],
            "local_metric_purity": dl["neighbor_purity"],
            "purity_gain": dl["neighbor_purity"] - de["neighbor_purity"],
            "local_condition": stats.mean_condition,
            "local_anisotropy": stats.mean_anisotropy,
        }
        rows.append(row)
        print(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
