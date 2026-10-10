from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--output", default="embedding_tsne.png")
    p.add_argument("--max_points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    d = np.load(args.features)
    z, y = d["z"], d["y"]
    if len(z) > args.max_points:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(z), args.max_points, replace=False)
        z, y = z[ids], y[ids]
    emb = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=args.seed).fit_transform(z)
    plt.figure(figsize=(8, 7))
    plt.scatter(emb[:, 0], emb[:, 1], c=y, s=5, alpha=0.7)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
