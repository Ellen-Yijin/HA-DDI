#!/usr/bin/env python3
"""
多种子实验流水线：自动训练 → 评估 → 汇总 mean±std → 显著性检验。

用法示例：
  python run_multiseed_experiment.py --task both
  python run_multiseed_experiment.py --task UnseenDDIs --seeds 42,123,456,789,2024
  python run_multiseed_experiment.py --skip_train --results_dir results/20250622_run1
  python run_multiseed_experiment.py --aggregate_only --results_dir results/20250622_run1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_CONFIGS = ["full", "wocon", "womv", "wocoattn"]

TASKS = {
    "UnseenDDIs": {
        "dir": "UnseenDDIs",
        "train": "Train_UnseenDDIs.py",
        "eval": "UnseenDDIs_full_evaluate.py",
    },
    "UnseenDrugs": {
        "dir": "UnseenDrugs",
        "train": "Train_UnseenDrugs.py",
        "eval": "UnseenDrugsnew_evaluate.py",
    },
}


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}  (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed train + eval + statistics pipeline")
    parser.add_argument("--task", choices=["UnseenDDIs", "UnseenDrugs", "both"], default="both")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--skip_train", action="store_true", help="Only run evaluation")
    parser.add_argument("--skip_eval", action="store_true", help="Only run training")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing raw_runs.csv")
    parser.add_argument("--fresh", action="store_true", help="Delete existing raw_runs.csv before a new run")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory for weights & CSV")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    classification_root = os.path.dirname(exp_dir)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(exp_dir, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)

    raw_csv = os.path.join(results_dir, "raw_runs.csv")
    meta_path = os.path.join(results_dir, "run_config.txt")

    if args.fresh and os.path.exists(raw_csv):
        os.remove(raw_csv)
        print(f"[Fresh] Removed existing {raw_csv}")

    if not args.aggregate_only:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"task={args.task}\n")
            f.write(f"seeds={seeds}\n")
            f.write(f"configs={configs}\n")
            f.write(f"skip_train={args.skip_train}\n")
            f.write(f"skip_eval={args.skip_eval}\n")

        tasks = list(TASKS.keys()) if args.task == "both" else [args.task]

        for task in tasks:
            info = TASKS[task]
            task_code_dir = os.path.join(classification_root, info["dir"])

            for seed in seeds:
                seed_weights_dir = os.path.join(results_dir, task, f"seed_{seed}")
                os.makedirs(seed_weights_dir, exist_ok=True)

                for config in configs:
                    banner = f"{task} | seed={seed} | config={config}"
                    print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")

                    if not args.skip_train:
                        train_cmd = [
                            args.python,
                            info["train"],
                            "--config",
                            config,
                            "--seed",
                            str(seed),
                            "--weights_dir",
                            seed_weights_dir,
                        ]
                        if task == "UnseenDrugs":
                            train_cmd.extend(["--val_scenario", "S2"])
                        run_cmd(train_cmd, cwd=task_code_dir)

                    if not args.skip_eval:
                        eval_cmd = [
                            args.python,
                            info["eval"],
                            "--config",
                            config,
                            "--seed",
                            str(seed),
                            "--weights_dir",
                            seed_weights_dir,
                            "--csv_path",
                            raw_csv,
                        ]
                        if task == "UnseenDrugs":
                            eval_cmd.extend([
                                "--train_val_scenario", "S2",
                                "--eval_scenarios", "S2,S3",
                                "--ablation_suite", "multiseed",
                                "--eval_setting", "coldstart",
                            ])
                        else:
                            eval_cmd.extend([
                                "--ablation_suite", "multiseed",
                                "--eval_setting", "transductive",
                            ])
                        run_cmd(eval_cmd, cwd=task_code_dir)

    if not args.skip_eval or args.aggregate_only:
        if not os.path.exists(raw_csv):
            print(f"[Error] raw_runs.csv not found: {raw_csv}")
            sys.exit(1)
        agg_script = os.path.join(exp_dir, "aggregate_statistics.py")
        run_cmd(
            [args.python, agg_script, "--input", raw_csv, "--output_dir", results_dir],
            cwd=exp_dir,
        )

    print(f"\n[Done] Results directory: {results_dir}")


if __name__ == "__main__":
    main()
