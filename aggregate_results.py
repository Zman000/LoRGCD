from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--metric", default="eval_all_acc")
    p.add_argument("--reduction", choices=["last", "best"], default="last",
                   help="Use last by default to avoid hidden-label checkpoint selection")
    args = p.parse_args()

    groups = defaultdict(list)
    for path in glob.glob(os.path.join(args.root, "**", "history.json"), recursive=True):
        with open(path) as f:
            hist = json.load(f)
        vals = [r[args.metric] for r in hist if args.metric in r]
        if not vals:
            continue
        value = vals[-1] if args.reduction == "last" else max(vals)
        rel = os.path.relpath(os.path.dirname(path), args.root)
        # Collapse final seed directory, preserving dataset/variant.
        parts = rel.split(os.sep)
        key = os.path.join(*parts[:-1]) if parts and parts[-1].startswith("seed_") else rel
        groups[key].append(value)

    for key in sorted(groups):
        a = np.asarray(groups[key], dtype=float)
        print(f"{key:45s} {a.mean()*100:6.2f} ± {a.std(ddof=1)*100 if len(a)>1 else 0:5.2f}  n={len(a)}")


if __name__ == "__main__":
    main()
