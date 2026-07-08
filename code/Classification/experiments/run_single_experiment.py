#!/usr/bin/env python3
"""
一次性完整消融：训练 + 评估（可视化请单独跑 case_study_vis.py）。

默认跑通：
  - 任务：UnseenDDIs + UnseenDrugs
  - 配置：full / wocon / womv / wocoattn
  - 三种评估场景：
      1. UnseenDDIs test          (transductive)
      2. UnseenDrugs unseenone    (cold-start S2)
      3. UnseenDrugs unseentwo    (cold-start S3)

断点续跑（默认开启）：
  - 已有 stage2 权重 → 跳过训练
  - 仅有 stage1 权重 → 只跑 stage2（Train_*.py 内 --resume）
  - all_results.csv 追加写入，已评估的 config 自动跳过
  - 全部完成后汇总 by_scenario / REPORT

用法：
  python run_single_experiment.py --seed 42 --results_dir /path/to/results
  python run_single_experiment.py --seed 42 --results_dir /path/to/results --no_resume
  python run_single_experiment.py --seed 42 --results_dir /path/to/results --skip_train
  python run_single_experiment.py --include_s3_training
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
    unseen_ddis_eval_done,
    unseen_drugs_eval_done,
)

ALL_CONFIGS = ["full", "wocon", "womv", "wocoattn"]
ABLATION_SUITE = "full_ablation_once"

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
    print(f"\n>>> {' '.join(cmd)}")
    print(f"    cwd: {cwd}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_unseen_ddis(
    python: str,
    classification_root: str,
    configs: list[str],
    seed: int,
    weights_dir: str,
    raw_csv: str,
    skip_train: bool,
    skip_eval: bool,
    resume: bool,
    force_train: bool,
) -> None:
    info = TASKS["UnseenDDIs"]
    task_dir = os.path.join(classification_root, info["dir"])
    os.makedirs(weights_dir, exist_ok=True)

    if resume and not skip_train:
        print_training_status(weights_dir, configs)

    for config in configs:
        banner = f"UnseenDDIs | transductive | config={config} | seed={seed}"
        print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")

        if not skip_train:
            if should_skip_training(weights_dir, config, resume=resume, force_train=force_train):
                print(f"[Resume] skip train (stage2 exists): {config}")
            else:
                status = training_status(weights_dir, config)
                if status == "stage1_only":
                    print(f"[Resume] continue stage2 only: {config}")
                train_cmd = [
                    python, info["train"],
                    "--config", config,
                    "--seed", str(seed),
                    "--weights_dir", weights_dir,
                ]
                if resume:
                    train_cmd.append("--resume")
                if force_train:
                    train_cmd.append("--force_train")
                run_cmd(train_cmd, cwd=task_dir)

        if not skip_eval:
            if resume and unseen_ddis_eval_done(raw_csv, seed, config, ABLATION_SUITE):
                print(f"[Resume] skip eval (already in CSV): UnseenDDIs/{config}")
            else:
                run_cmd(
                    [
                        python, info["eval"],
                        "--config", config,
                        "--seed", str(seed),
                        "--weights_dir", weights_dir,
                        "--csv_path", raw_csv,
                        "--ablation_suite", ABLATION_SUITE,
                        "--eval_setting", "transductive",
                    ],
                    cwd=task_dir,
                )


def run_unseen_drugs_block(
    python: str,
    classification_root: str,
    configs: list[str],
    seed: int,
    weights_root: str,
    raw_csv: str,
    train_val_scenario: str,
    skip_train: bool,
    skip_eval: bool,
    resume: bool,
    force_train: bool,
) -> None:
    info = TASKS["UnseenDrugs"]
    task_dir = os.path.join(classification_root, info["dir"])
    weights_dir = os.path.join(weights_root, f"val{train_val_scenario}")
    os.makedirs(weights_dir, exist_ok=True)

    if resume and not skip_train:
        print_training_status(weights_dir, configs)

    for config in configs:
        banner = (
            f"UnseenDrugs | train_val={train_val_scenario} | "
            f"eval S2+S3 | config={config} | seed={seed}"
        )
        print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}")

        if not skip_train:
            if should_skip_training(weights_dir, config, resume=resume, force_train=force_train):
                print(f"[Resume] skip train (stage2 exists): {config}")
            else:
                status = training_status(weights_dir, config)
                if status == "stage1_only":
                    print(f"[Resume] continue stage2 only: {config}")
                train_cmd = [
                    python, info["train"],
                    "--config", config,
                    "--seed", str(seed),
                    "--weights_dir", weights_dir,
                    "--val_scenario", train_val_scenario,
                ]
                if resume:
                    train_cmd.append("--resume")
                if force_train:
                    train_cmd.append("--force_train")
                run_cmd(train_cmd, cwd=task_dir)

        if not skip_eval:
            if resume and unseen_drugs_eval_done(
                raw_csv, seed, config, ABLATION_SUITE, train_val_scenario, ("S2", "S3")
            ):
                print(f"[Resume] skip eval (already in CSV): UnseenDrugs/{config}")
            else:
                run_cmd(
                    [
                        python, info["eval"],
                        "--config", config,
                        "--seed", str(seed),
                        "--weights_dir", weights_dir,
                        "--csv_path", raw_csv,
                        "--train_val_scenario", train_val_scenario,
                        "--eval_scenarios", "S2,S3",
                        "--ablation_suite", ABLATION_SUITE,
                        "--eval_setting", "coldstart",
                    ],
                    cwd=task_dir,
                )


def export_results(python: str, exp_dir: str, results_dir: str, raw_csv: str) -> None:
    if not os.path.exists(raw_csv):
        print(f"[Warn] No results CSV at {raw_csv}, skip export.")
        return
    export_script = os.path.join(exp_dir, "export_results_by_scenario.py")
    run_cmd(
        [
            python, export_script,
            "--input", raw_csv,
            "--output_dir", results_dir,
        ],
        cwd=exp_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot full ablation: train+eval on 3 scenarios (transductive + cold-start S2/S3)"
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
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable checkpoint resume (retrain all; still appends CSV unless --fresh_results)",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Retrain even when stage2 weights exist (overwrites checkpoints)",
    )
    parser.add_argument(
        "--fresh_results",
        action="store_true",
        help="Delete all_results.csv before run (default: append / skip existing eval rows)",
    )
    parser.add_argument(
        "--include_s3_training",
        action="store_true",
        help="Also train UnseenDrugs with S3 validation (extra 4 runs); default only S2 val training",
    )
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    resume = not args.no_resume

    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    for c in configs:
        if c not in ALL_CONFIGS:
            parser.error(f"Unknown config: {c}")

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    classification_root = os.path.dirname(exp_dir)

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    else:
        results_dir = os.path.join(
            exp_dir, "results", "full_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(results_dir, exist_ok=True)

    raw_csv = os.path.join(results_dir, "all_results.csv")
    if args.fresh_results and os.path.exists(raw_csv):
        os.remove(raw_csv)
        print(f"[Init] Removed existing {raw_csv}")

    meta_path = os.path.join(results_dir, "run_info.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("mode=full_ablation_once\n")
        f.write(f"configs={configs}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"resume={resume}\n")
        f.write(f"force_train={args.force_train}\n")
        f.write(f"include_s3_training={args.include_s3_training}\n")
        f.write("scenarios=UnseenDDIs_test,UnseenDrugs_S2,UnseenDrugs_S3\n")
        f.write(f"results_dir={results_dir}\n")

    print("=" * 70)
    print("Full ablation once — train + eval per config, aggregate at end")
    print(f"  configs     : {configs}")
    print(f"  seed        : {args.seed}")
    print(f"  output      : {results_dir}")
    print(f"  resume      : {resume}")
    print(f"  force_train : {args.force_train}")
    print("  flow        : train one config → eval one config → ... → export")
    print("  scenarios:")
    print("    [1] UnseenDDIs test (transductive)")
    print("    [2] UnseenDrugs unseenone (cold-start S2)")
    print("    [3] UnseenDrugs unseentwo (cold-start S3)")
    print("=" * 70)

    # --- 1) UnseenDDIs transductive ---
    run_unseen_ddis(
        args.python,
        classification_root,
        configs,
        args.seed,
        os.path.join(results_dir, "UnseenDDIs"),
        raw_csv,
        args.skip_train,
        args.skip_eval,
        resume,
        args.force_train,
    )

    # --- 2) UnseenDrugs cold-start (train val S2, eval S2+S3) ---
    run_unseen_drugs_block(
        args.python,
        classification_root,
        configs,
        args.seed,
        os.path.join(results_dir, "UnseenDrugs"),
        raw_csv,
        train_val_scenario="S2",
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        resume=resume,
        force_train=args.force_train,
    )

    # --- 3) Optional: UnseenDrugs train with S3 val ---
    if args.include_s3_training:
        run_unseen_drugs_block(
            args.python,
            classification_root,
            configs,
            args.seed,
            os.path.join(results_dir, "UnseenDrugs"),
            raw_csv,
            train_val_scenario="S3",
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            resume=resume,
            force_train=args.force_train,
        )

    if not args.skip_eval:
        export_results(args.python, exp_dir, results_dir, raw_csv)

    print("\n" + "=" * 70)
    print("[Done] Full ablation run finished.")
    print(f"  Weights   : {results_dir}/UnseenDDIs/  &  {results_dir}/UnseenDrugs/valS2/")
    print(f"  All metrics : {results_dir}/all_results.csv")
    print(f"  By scenario : {results_dir}/by_scenario/*.csv")
    print(f"  Report      : {results_dir}/REPORT_BY_SCENARIO.md")
    print("  Visualization: run case_study_vis.py separately when needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
