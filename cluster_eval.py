from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

from metrics import evaluate_gcd


def main():
    p = argparse.ArgumentParser("Cluster extracted embeddings with known/estimated K")
    p.add_argument("--features", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_init", type=int, default=20)
    p.add_argument("--unlabeled_only", action="store_true", default=True)
    args = p.parse_args()

    d = np.load(args.features)
    z, y = d["z"], d["y"]
    mask = d["unlabeled"].astype(bool) if "unlabeled" in d else np.ones(len(y), bool)
    z, y = z[mask], y[mask]
    old = d["old_classes"] if "old_classes" in d else np.arange(args.k // 2)
    pred = KMeans(n_clusters=args.k, n_init=args.n_init, random_state=args.seed).fit_predict(z)
    metrics, _ = evaluate_gcd(y, pred, old)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
