#!/usr/bin/env python3
"""
仅跑 UnseenDrugs full 在 cold-start inductive 场景 S2 + S3 的训练与评估。

默认流程：
  1. 用 S2（unseen one drug）作验证集训练 full 模型（early stopping / checkpoint）
  2. 在 S2、S3 两个 inductive 验证集上评估并写入 CSV

用法：
  python run_unseen_drugs_full_inductive.py --seed 42 --results_dir /path/to/results

  # 已有权重，只评估
  python run_unseen_drugs_full_inductive.py --seed 42 --results_dir ... --skip_train

  # 强制重训
  python run_unseen_drugs_full_inductive.py --seed 42 --results_dir ... --force_train
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
    unseen_drugs_eval_done,
)

CONFIG = "full"
ABLATION_SUITE = "unseen_drugs_full_inductive"
TRAIN_VAL_SCENARIO = "S2"
EVAL_SCENARIOS = ("S2", "S3")


def run_cmd(cmd: list[str], cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    cwd: {cwd}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train+eval UnseenDrugs full on cold-start S2/S3 (inductive)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Output root (weights under UnseenDrugs/valS2/, CSV at all_results.csv)",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    resume = not args.no_resume
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    classification_root = os.path.dirname(exp_dir)
    task_dir = os.path.join(classification_root, "UnseenDrugs")

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(
            exp_dir,
            "results",
            "unseen_drugs_full_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    os.makedirs(results_dir, exist_ok=True)

    weights_dir = os.path.join(results_dir, "UnseenDrugs", f"val{TRAIN_VAL_SCENARIO}")
    os.makedirs(weights_dir, exist_ok=True)
    raw_csv = os.path.join(results_dir, "all_results.csv")

    meta_path = os.path.join(results_dir, "run_info.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("mode=unseen_drugs_full_inductive\n")
        f.write(f"config={CONFIG}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"train_val_scenario={TRAIN_VAL_SCENARIO}\n")
        f.write(f"eval_scenarios={','.join(EVAL_SCENARIOS)}\n")
        f.write(f"resume={resume}\n")
        f.write(f"results_dir={results_dir}\n")

    print("=" * 70)
    print("UnseenDrugs full — cold-start inductive (S2 train-val, eval S2+S3)")
    print(f"  config          : {CONFIG}")
    print(f"  seed            : {args.seed}")
    print(f"  train val split : {TRAIN_VAL_SCENARIO} (unseen one drug)")
    print(f"  eval scenarios  : {', '.join(EVAL_SCENARIOS)}")
    print(f"  weights_dir     : {weights_dir}")
    print(f"  results_csv     : {raw_csv}")
    print(f"  resume          : {resume}")
    print("=" * 70)

    if resume and not args.skip_train:
        print_training_status(weights_dir, [CONFIG])

    # --- train ---
    if not args.skip_train:
        if should_skip_training(weights_dir, CONFIG, resume=resume, force_train=args.force_train):
            print(f"[Resume] skip train (stage2 exists): {CONFIG}")
        else:
            train_cmd = [
                args.python,
                "Train_UnseenDrugs.py",
                "--config",
                CONFIG,
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

    # --- eval S2 + S3 ---
    if not args.skip_eval:
        if resume and unseen_drugs_eval_done(
            raw_csv, args.seed, CONFIG, ABLATION_SUITE, TRAIN_VAL_SCENARIO, EVAL_SCENARIOS
        ):
            print(f"[Resume] skip eval (S2+S3 already in CSV)")
        else:
            run_cmd(
                [
                    args.python,
                    "UnseenDrugsnew_evaluate.py",
                    "--config",
                    CONFIG,
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

            export_script = os.path.join(exp_dir, "export_results_by_scenario.py")
            if os.path.isfile(export_script):
                run_cmd(
                    [
                        args.python,
                        export_script,
                        "--input",
                        raw_csv,
                        "--output_dir",
                        results_dir,
                    ],
                    cwd=exp_dir,
                )

    print("\n" + "=" * 70)
    print("[Done] UnseenDrugs full inductive run finished.")
    print(f"  Weights : {weights_dir}/weights_{CONFIG}_stage2.weights.h5")
    print(f"  Metrics : {raw_csv}")
    print(f"  Report  : {results_dir}/by_scenario/ (if export ran)")
    print("=" * 70)


if __name__ == "__main__":
    main()
