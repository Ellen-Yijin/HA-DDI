#!/usr/bin/env python3
"""
Cold-start（S2/S3 inductive）专项消融：对比学习 / 多视图分支。

背景：在 transductive 测试上 wocon/womv 掉点往往很小；在 S2（unseen one drug）
与 S3（unseen two drugs）上对比分支贡献通常更明显，更适合支撑 cold-start 鲁棒性论述。

默认流程（UnseenDrugs）：
  - 配置：full / wocon（去对比）/ womv（去多视图）
  - 训练验证：S2 与 S3 各跑一遍（可用 --train_val_scenarios 控制）
  - 评估：在 S2 + S3 两个 inductive 集上同时报告指标

用法：
  python run_coldstart_ablation.py
  python run_coldstart_ablation.py --seeds 42,123,456 --train_val_scenarios S2
  python run_coldstart_ablation.py --configs full,wocon --skip_train
  python run_coldstart_ablation.py --aggregate_only --results_dir results/coldstart_xxx
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

from scenario_utils import COLDSTART_DEFAULT_CONFIGS

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_TRAIN_VAL = ["S2", "S3"]
ABLATION_SUITE = "coldstart_contrastive"

TASK = "UnseenDrugs"
TASK_DIR = "UnseenDrugs"
TRAIN_SCRIPT = "Train_UnseenDrugs.py"
EVAL_SCRIPT = "UnseenDrugsnew_evaluate.py"


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    cwd: {cwd}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start S2/S3 ablation (contrastive / multi-view)")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(COLDSTART_DEFAULT_CONFIGS),
        help="Ablation configs (default: full,wocon,womv)",
    )
    parser.add_argument(
        "--train_val_scenarios",
        type=str,
        default=",".join(DEFAULT_TRAIN_VAL),
        help="Training validation splits: S2, S3, or S2,S3 (default both)",
    )
    parser.add_argument(
        "--eval_scenarios",
        type=str,
        default="S2,S3",
        help="Eval inductive splits (default: S2,S3)",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Remove existing raw_runs.csv before run")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    classification_root = os.path.dirname(exp_dir)
    task_code_dir = os.path.join(classification_root, TASK_DIR)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    train_val_scenarios = [s.strip() for s in args.train_val_scenarios.split(",") if s.strip()]
    for s in train_val_scenarios:
        if s not in {"S2", "S3"}:
            parser.error(f"Invalid train_val_scenario: {s}")

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(exp_dir, "results", "coldstart_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)

    raw_csv = os.path.join(results_dir, "raw_runs.csv")
    meta_path = os.path.join(results_dir, "run_config.txt")

    if args.fresh and os.path.exists(raw_csv):
        os.remove(raw_csv)
        print(f"[Fresh] Removed {raw_csv}")

    if not args.aggregate_only:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"suite={ABLATION_SUITE}\n")
            f.write(f"task={TASK}\n")
            f.write(f"seeds={seeds}\n")
            f.write(f"configs={configs}\n")
            f.write(f"train_val_scenarios={train_val_scenarios}\n")
            f.write(f"eval_scenarios={args.eval_scenarios}\n")

        for train_val in train_val_scenarios:
            for seed in seeds:
                seed_weights_dir = os.path.join(results_dir, TASK, f"val{train_val}", f"seed_{seed}")
                os.makedirs(seed_weights_dir, exist_ok=True)

                for config in configs:
                    banner = f"{TASK} | val={train_val} | seed={seed} | config={config}"
                    print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")

                    if not args.skip_train:
                        run_cmd(
                            [
                                args.python,
                                TRAIN_SCRIPT,
                                "--config",
                                config,
                                "--seed",
                                str(seed),
                                "--weights_dir",
                                seed_weights_dir,
                                "--val_scenario",
                                train_val,
                            ],
                            cwd=task_code_dir,
                        )

                    if not args.skip_eval:
                        run_cmd(
                            [
                                args.python,
                                EVAL_SCRIPT,
                                "--config",
                                config,
                                "--seed",
                                str(seed),
                                "--weights_dir",
                                seed_weights_dir,
                                "--train_val_scenario",
                                train_val,
                                "--eval_scenarios",
                                args.eval_scenarios,
                                "--ablation_suite",
                                ABLATION_SUITE,
                                "--eval_setting",
                                "coldstart",
                                "--csv_path",
                                raw_csv,
                            ],
                            cwd=task_code_dir,
                        )

    if not args.skip_eval or args.aggregate_only:
        if not os.path.exists(raw_csv):
            print(f"[Error] raw_runs.csv not found: {raw_csv}")
            sys.exit(1)
        agg_script = os.path.join(exp_dir, "aggregate_coldstart_ablation.py")
        run_cmd(
            [
                args.python,
                agg_script,
                "--input",
                raw_csv,
                "--output_dir",
                results_dir,
            ],
            cwd=exp_dir,
        )

    print(f"\n[Done] Cold-start ablation results: {results_dir}")
    print("  raw_runs.csv          — 全部 run 指标")
    print("  coldstart_summary.csv — S2/S3 mean±std")
    print("  coldstart_significance.csv — full vs wocon/womv 显著性")
    print("  REPORT_COLDSTART.md   — 论文用表格摘要")


if __name__ == "__main__":
    main()
