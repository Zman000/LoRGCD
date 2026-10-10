from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser("Launch the core CVPR ablation grid")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--split_file", default=None)
    p.add_argument("--base_output", default="runs/ablations")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--backbone", default="vit_base_patch16_224.dino")
    p.add_argument("--seeds", default="42,17,123")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    variants = {
        "baseline": ["--lambda_rad", "0", "--lambda_gmd", "0"],
        "rad_only": ["--lambda_rad", "0.5", "--lambda_gmd", "0"],
        "gmd_only": ["--lambda_rad", "0", "--lambda_gmd", "0.5"],
        "full": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5"],
        "euclidean_graph": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5", "--geometry_mode", "euclidean"],
        "global_mahalanobis": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5", "--geometry_mode", "global"],
        "local_no_tangent_bundle": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5", "--disable_tangent_alignment"],
        "fixed_tangent_rank": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5", "--fixed_tangent_rank"],
        "no_labeled_anchors": ["--lambda_rad", "0.5", "--lambda_gmd", "0.5", "--disable_supervised_anchors"],
    }

    for (name, extra), seed in itertools.product(variants.items(), seeds):
        out = os.path.join(args.base_output, args.dataset, name, f"seed_{seed}")
        cmd = [
            "python", "train.py", "--dataset", args.dataset, "--data_root", args.data_root,
            "--epochs", str(args.epochs), "--backbone", args.backbone,
            "--seed", str(seed), "--output", out, "--amp",
        ]
        if args.split_file:
            cmd += ["--split_file", args.split_file]
        cmd += extra
        print(" ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
