from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess


def _csv_numbers(text, cast=float):
    return [cast(v.strip()) for v in text.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser("Launch theory-driven LoRGCD structural sweeps")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_file", default=None)
    p.add_argument("--backbone", default="vit_base_patch16_224.dino")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", default="runs/structural_sweeps")
    p.add_argument("--sweep", choices=["k", "rank", "rho", "steps", "label_fraction"], required=True)
    p.add_argument("--k_values", default="8,16,32,64")
    p.add_argument("--rank_values", default="2,4,8,16")
    p.add_argument("--rho_values", default="0.03,0.1,0.3,1.0")
    p.add_argument("--step_values", default="1,2,3")
    p.add_argument("--label_values", default="0.1,0.25,0.5")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    specs = {
        "k": ("geo_k", _csv_numbers(args.k_values, int)),
        "rank": ("geo_rank", _csv_numbers(args.rank_values, int)),
        "rho": ("geo_shrinkage", _csv_numbers(args.rho_values, float)),
        "steps": ("geo_steps", _csv_numbers(args.step_values, int)),
        "label_fraction": ("labeled_fraction", _csv_numbers(args.label_values, float)),
    }
    flag, values = specs[args.sweep]

    for value in values:
        tag = str(value).replace(".", "p")
        out = os.path.join(args.output_root, args.dataset, args.sweep, tag, f"seed_{args.seed}")
        cmd = [
            "python", "train.py", "--dataset", args.dataset, "--data_root", args.data_root,
            "--backbone", args.backbone, "--epochs", str(args.epochs), "--seed", str(args.seed),
            "--output", out, "--amp", "--" + flag, str(value),
        ]
        if args.split_file:
            cmd += ["--split_file", args.split_file]
        print(" ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
