#!/usr/bin/env python3
"""
UnseenDrugs 冷启动 inductive 消融：训练 + 评估（S2/S3）。

默认跑 4 个消融配置：full / wocon / womv / wocoattn
  - 训练验证集：S2（unseen one drug）
  - 评估场景：S2 + S3（inductive）

流程（每个 config）：
  train one → eval one（S2+S3）→ 下一个 config → 最后汇总

断点续跑（默认开启）：
  - 已有 stage2 权重 → 跳过训练
  - CSV 已有该 config 的 S2+S3 结果 → 跳过评估

用法：
  python run_unseen_drugs_ablation_inductive.py --seed 42 --results_dir /path/to/results

  # 只跑剩余消融（跳过已完成的 full）
  python run_unseen_drugs_ablation_inductive.py --seed 42 --results_dir ... --config wocon,womv,wocoattn

  # 只评估
  python run_unseen_drugs_ablation_inductive.py --seed 42 --results_dir ... --skip_train
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

from checkpoint_utils import (
    print_training_status,
    should_skip_training,
    training_status,
    unseen_drugs_eval_done,
)

ALL_CONFIGS = ["full", "wocon", "womv", "wocoattn"]
ABLATION_SUITE = "unseen_drugs_ablation_inductive"
TRAIN_VAL_SCENARIO = "S2"
EVAL_SCENARIOS = ("S2", "S3")

TRAIN_SCRIPT = "Train_UnseenDrugs.py"
EVAL_SCRIPT = "UnseenDrugsnew_evaluate.py"


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    cwd: {cwd}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def export_results(python: str, exp_dir: str, results_dir: str, raw_csv: str) -> None:
    if not os.path.isfile(raw_csv):
        print(f"[Warn] No results CSV at {raw_csv}, skip export.")
        return
    export_script = os.path.join(exp_dir, "export_results_by_scenario.py")
    if not os.path.isfile(export_script):
        print(f"[Warn] export script not found: {export_script}")
        return
    run_cmd(
        [
            python,
            export_script,
            "--input",
            raw_csv,
            "--output_dir",
            results_dir,
        ],
        cwd=exp_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UnseenDrugs ablation on cold-start inductive S2/S3 (train val=S2, eval S2+S3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=",".join(ALL_CONFIGS),
        help="Comma-separated configs (default: full,wocon,womv,wocoattn)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument(
        "--fresh_results",
        action="store_true",
        help="Delete all_results.csv before run (default: append / skip existing eval rows)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Output root (weights: UnseenDrugs/valS2/, CSV: all_results.csv)",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    resume = not args.no_resume
    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    for c in configs:
        if c not in ALL_CONFIGS:
            parser.error(f"Unknown config: {c}")

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    classification_root = os.path.dirname(exp_dir)
    task_dir = os.path.join(classification_root, "UnseenDrugs")

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(
            exp_dir,
            "results",
            "unseen_drugs_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    os.makedirs(results_dir, exist_ok=True)

    weights_dir = os.path.join(results_dir, "UnseenDrugs", f"val{TRAIN_VAL_SCENARIO}")
    os.makedirs(weights_dir, exist_ok=True)
    raw_csv = os.path.join(results_dir, "all_results.csv")

    if args.fresh_results and os.path.isfile(raw_csv):
        os.remove(raw_csv)
        print(f"[Init] Removed existing {raw_csv}")

    meta_path = os.path.join(results_dir, "run_info.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("mode=unseen_drugs_ablation_inductive\n")
        f.write(f"configs={configs}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"train_val_scenario={TRAIN_VAL_SCENARIO}\n")
        f.write(f"eval_scenarios={','.join(EVAL_SCENARIOS)}\n")
        f.write(f"resume={resume}\n")
        f.write(f"force_train={args.force_train}\n")
        f.write(f"results_dir={results_dir}\n")

    print("=" * 70)
    print("UnseenDrugs ablation — cold-start inductive (train val=S2, eval S2+S3)")
    print(f"  configs         : {configs}")
    print(f"  seed            : {args.seed}")
    print(f"  train val split : {TRAIN_VAL_SCENARIO} (unseen one drug)")
    print(f"  eval scenarios  : {', '.join(EVAL_SCENARIOS)}")
    print(f"  weights_dir     : {weights_dir}")
    print(f"  results_csv     : {raw_csv}")
    print(f"  resume          : {resume}")
    print(f"  flow            : per config: train → eval → next")
    print("=" * 70)

    if resume and not args.skip_train:
        print_training_status(weights_dir, configs)

    for config in configs:
        banner = (
            f"UnseenDrugs | inductive S2+S3 | train_val={TRAIN_VAL_SCENARIO} | "
            f"config={config} | seed={args.seed}"
        )
        print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")

        if not args.skip_train:
            if should_skip_training(weights_dir, config, resume=resume, force_train=args.force_train):
                print(f"[Resume] skip train (stage2 exists): {config}")
            else:
                if training_status(weights_dir, config) == "stage1_only":
                    print(f"[Resume] continue stage2 only: {config}")
                train_cmd = [
                    args.python,
                    TRAIN_SCRIPT,
                    "--config",
                    config,
                    "--seed",
                    str(args.seed),
                    "--weights_dir",
                    weights_dir,
                    "--val_scenario",
                    TRAIN_VAL_SCENARIO,
                ]
                if resume:
                    train_cmd.append("--resume")
                if args.force_train:
                    train_cmd.append("--force_train")
                run_cmd(train_cmd, cwd=task_dir)

        if not args.skip_eval:
            if resume and unseen_drugs_eval_done(
                raw_csv,
                args.seed,
                config,
                ABLATION_SUITE,
                TRAIN_VAL_SCENARIO,
                EVAL_SCENARIOS,
            ):
                print(f"[Resume] skip eval (S2+S3 already in CSV): {config}")
            else:
                run_cmd(
                    [
                        args.python,
                        EVAL_SCRIPT,
                        "--config",
                        config,
                        "--seed",
                        str(args.seed),
                        "--weights_dir",
                        weights_dir,
                        "--csv_path",
                        raw_csv,
                        "--train_val_scenario",
                        TRAIN_VAL_SCENARIO,
                        "--eval_scenarios",
                        ",".join(EVAL_SCENARIOS),
                        "--ablation_suite",
                        ABLATION_SUITE,
                        "--eval_setting",
                        "coldstart",
                    ],
                    cwd=task_dir,
                )

    if not args.skip_eval:
        export_results(args.python, exp_dir, results_dir, raw_csv)

    print("\n" + "=" * 70)
    print("[Done] UnseenDrugs ablation inductive run finished.")
    print(f"  Weights   : {weights_dir}/")
    print(f"  Metrics   : {raw_csv}")
    print(f"  By scenario : {results_dir}/by_scenario/")
    print(f"  Report      : {results_dir}/REPORT_BY_SCENARIO.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
