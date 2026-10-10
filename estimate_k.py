from __future__ import annotations

import argparse
import json

import numpy as np

from geometry import estimate_num_classes_spectral


def main():
    p = argparse.ArgumentParser("LoRGCD spectral category-count estimator")
    p.add_argument("--features", required=True, help="features_eval.npz or any npz containing z")
    p.add_argument("--min_k", type=int, required=True, help="normally number of known classes")
    p.add_argument("--max_k", type=int, required=True)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--shrinkage", type=float, default=0.10)
    p.add_argument("--candidate_factor", type=int, default=4,
                   help="Cosine candidate pool multiplier before tangent-bundle re-ranking")
    p.add_argument("--disable_tangent_alignment", action="store_true")
    p.add_argument("--fixed_tangent_rank", action="store_true",
                   help="Use one fixed tangent rank instead of participation-ratio adaptive rank")
    p.add_argument("--unlabeled_only", action="store_true")
    args = p.parse_args()

    data = np.load(args.features)
    z = data["z"]
    if args.unlabeled_only and "unlabeled" in data:
        z = z[data["unlabeled"].astype(bool)]
    k_hat, eig = estimate_num_classes_spectral(
        z, args.min_k, args.max_k, args.neighbors, args.rank, args.shrinkage,
        args.candidate_factor, not args.disable_tangent_alignment, not args.fixed_tangent_rank
    )
    print(json.dumps({"estimated_k": int(k_hat), "eigenvalues": eig.tolist()}, indent=2))


if __name__ == "__main__":
    main()
