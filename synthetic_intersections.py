from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from geometry import local_graph_diagnostics, local_riemannian_affinity


def make_near_intersecting_lines(
    angle: float,
    separation: float,
    per_class: int = 200,
    dim: int = 16,
    noise: float = 0.01,
    seed: int = 0,
):
    """Two locally linear class manifolds with small positional separation.

    A large common base vector makes cosine distance behave like a local ambient metric.
    The classes differ in tangent orientation by `angle` radians. This is a controlled
    test of the Grassmann term rather than a natural-image benchmark.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(dim, dtype=np.float64)
    base[0] = 4.0
    d1 = np.zeros(dim, dtype=np.float64)
    d1[1] = 1.0
    d2 = np.zeros(dim, dtype=np.float64)
    d2[1] = np.cos(angle)
    d2[2] = np.sin(angle)
    o1 = np.zeros(dim, dtype=np.float64)
    o2 = np.zeros(dim, dtype=np.float64)
    o1[3], o2[3] = separation / 2.0, -separation / 2.0
    t = np.linspace(-1.0, 1.0, per_class)
    x1 = base + o1 + t[:, None] * d1 + noise * rng.normal(size=(per_class, dim))
    x2 = base + o2 + t[:, None] * d2 + noise * rng.normal(size=(per_class, dim))
    x = np.concatenate([x1, x2], axis=0).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    y = np.concatenate([np.zeros(per_class, np.int64), np.ones(per_class, np.int64)])
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    p = argparse.ArgumentParser("Synthetic validation of tangent-bundle consistency")
    p.add_argument("--angles", default="0.3,0.6,1.0,1.3,1.57", help="radians")
    p.add_argument("--separation", type=float, default=0.03)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--output", default="synthetic_intersections.csv")
    args = p.parse_args()

    rows = []
    for angle in [float(v) for v in args.angles.split(",")]:
        z, y = make_near_intersecting_lines(angle, args.separation, seed=123)
        a_m, _ = local_riemannian_affinity(
            z, k=args.k, rank=args.rank, use_local_metric=True, use_tangent_alignment=False
        )
        a_tb, stats = local_riemannian_affinity(
            z, k=args.k, rank=args.rank, use_local_metric=True, use_tangent_alignment=True
        )
        m = local_graph_diagnostics(a_m, y)
        tb = local_graph_diagnostics(a_tb, y)
        row = {
            "angle_rad": angle,
            "local_mahalanobis_purity": m["neighbor_purity"],
            "tangent_bundle_purity": tb["neighbor_purity"],
            "purity_gain": tb["neighbor_purity"] - m["neighbor_purity"],
            "mean_tangent_mismatch": stats.mean_tangent_mismatch,
        }
        rows.append(row)
        print(row)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
