from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import yaml


def main():
    p = argparse.ArgumentParser("Run a multi-dataset LoRGCD paper suite from YAML")
    p.add_argument("--config", default="configs/cvpr_suite.yaml")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    common = cfg.get("common", {})
    for ds in cfg["datasets"]:
        for seed in cfg.get("seeds", [42, 17, 123]):
            out = os.path.join(cfg.get("output_root", "runs/cvpr_suite"), ds["name"], f"seed_{seed}")
            cmd = ["python", "train.py", "--dataset", ds["name"], "--data_root", ds["root"],
                   "--seed", str(seed), "--output", out]
            merged = {**common, **{k: v for k, v in ds.items() if k not in {"name", "root"}}}
            for k, v in merged.items():
                flag = "--" + k
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                elif v is not None:
                    cmd.extend([flag, str(v)])
            print(" ".join(shlex.quote(str(x)) for x in cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
