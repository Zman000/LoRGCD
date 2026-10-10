from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser("Aggregate final-epoch/evaluation JSON into paper-ready CSV")
    p.add_argument("--root", required=True)
    p.add_argument("--output", default="paper_results.csv")
    args = p.parse_args()

    rows = []
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(args.root, "**", "evaluation.json"), recursive=True):
        with open(path) as f:
            r = json.load(f)
        rel = os.path.relpath(os.path.dirname(path), args.root)
        parts = rel.split(os.sep)
        key = os.path.join(*parts[:-1]) if parts and parts[-1].startswith("seed_") else rel
        groups[key].append(r)

    metrics = ["all_acc", "old_acc", "new_acc", "nmi", "ari"]
    for key, vals in sorted(groups.items()):
        row = {"experiment": key, "n": len(vals)}
        for m in metrics:
            a = np.array([v[m] for v in vals if m in v], dtype=float)
            row[m + "_mean"] = float(a.mean()) if len(a) else np.nan
            row[m + "_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rows.append(row)

    if not rows:
        raise SystemExit("No evaluation.json files found. Run evaluate.py on final checkpoints first.")
    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
