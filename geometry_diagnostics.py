from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from geometry import local_graph_diagnostics, local_riemannian_affinity


def main():
    p = argparse.ArgumentParser("Post-hoc geometry diagnostics on extracted features")
    p.add_argument("--features", required=True)
    p.add_argument("--max_points", type=int, default=512)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--shrinkage", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = np.load(args.features)
    z, y = d["z"], d["y"]
    mask = d["unlabeled"].astype(bool) if "unlabeled" in d else np.ones(len(y), dtype=bool)
    z, y = z[mask], y[mask]
    if len(z) > args.max_points:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(z), args.max_points, replace=False)
        z, y = z[ids], y[ids]
    zt = torch.from_numpy(z).float()
    yt = torch.from_numpy(y).long()

    variants = {}
    for name, local, tangent in [
        ("cosine", False, False),
        ("local_mahalanobis", True, False),
        ("tangent_bundle", True, True),
    ]:
        a, stats = local_riemannian_affinity(
            zt, k=args.k, rank=args.rank, shrinkage=args.shrinkage,
            use_local_metric=local, use_tangent_alignment=tangent,
        )
        diag = local_graph_diagnostics(a, yt)
        variants[name] = {
            **diag,
            "mean_condition": stats.mean_condition,
            "mean_anisotropy": stats.mean_anisotropy,
            "mean_tangent_mismatch": stats.mean_tangent_mismatch,
            "mean_tangent_rank": stats.mean_tangent_rank,
        }
    print(json.dumps(variants, indent=2))


if __name__ == "__main__":
    main()
