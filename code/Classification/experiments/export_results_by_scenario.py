#!/usr/bin/env python3
"""将 all_results.csv 按评估场景拆分，并生成汇总报告。"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd

from scenario_utils import DATASET_TO_SCENARIO

METRICS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_aupr"]

# (task, dataset) -> (file_slug, eval_setting, scenario_label)
SCENARIO_SPECS = {
    ("UnseenDDIs", "test"): {
        "file_slug": "01_UnseenDDIs_transductive_test",
        "eval_setting": "transductive",
        "scenario_label": "test (transductive)",
        "title": "UnseenDDIs — Transductive Test",
    },
    ("UnseenDrugs", "unseenone"): {
        "file_slug": "02_UnseenDrugs_coldstart_S2_unseenone",
        "eval_setting": "coldstart",
        "scenario_label": "S2 (unseen one drug)",
        "title": "UnseenDrugs — Cold-Start S2 (Unseen One Drug)",
    },
    ("UnseenDrugs", "unseentwo"): {
        "file_slug": "03_UnseenDrugs_coldstart_S3_unseentwo",
        "eval_setting": "coldstart",
        "scenario_label": "S3 (unseen two drugs)",
        "title": "UnseenDrugs — Cold-Start S3 (Unseen Two Drugs)",
    },
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "task" not in df.columns:
        df["task"] = "Unknown"
    if "seed" not in df.columns:
        df["seed"] = 42
    if "scenario" not in df.columns and "dataset" in df.columns:
        df["scenario"] = df["dataset"].map(lambda x: DATASET_TO_SCENARIO.get(x, x))
    if "eval_setting" not in df.columns:
        df["eval_setting"] = df.apply(_infer_eval_setting, axis=1)
    if "ablation_suite" not in df.columns:
        df["ablation_suite"] = "full_ablation_once"
    if "train_val_scenario" not in df.columns:
        df["train_val_scenario"] = df["task"].map(
            lambda t: "S2" if str(t) == "UnseenDrugs" else "n/a"
        )
    return df


def _infer_eval_setting(row) -> str:
    task = str(row.get("task", ""))
    dataset = str(row.get("dataset", ""))
    key = (task, dataset)
    if key in SCENARIO_SPECS:
        return SCENARIO_SPECS[key]["eval_setting"]
    return "unknown"


def scenario_key(row) -> Optional[Tuple[str, str]]:
    return str(row.get("task", "")), str(row.get("dataset", ""))


def export_by_scenario(df: pd.DataFrame, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    by_dir = os.path.join(output_dir, "by_scenario")
    os.makedirs(by_dir, exist_ok=True)

    exported = {}
    for key, spec in SCENARIO_SPECS.items():
        task, dataset = key
        sub = df[(df["task"] == task) & (df["dataset"] == dataset)].copy()
        if sub.empty:
            continue
        path = os.path.join(by_dir, f"{spec['file_slug']}.csv")
        sub.to_csv(path, index=False)
        exported[key] = {"path": path, "spec": spec, "df": sub}

    return exported


def build_summary_markdown(df: pd.DataFrame, exported: dict, results_dir: str) -> str:
    lines = [
        "# Full Ablation Run — Results by Scenario",
        "",
        f"Results directory: `{results_dir}`",
        "",
        "Three evaluation scenarios:",
        "1. **UnseenDDIs / test** — transductive",
        "2. **UnseenDrugs / unseenone** — cold-start S2",
        "3. **UnseenDrugs / unseentwo** — cold-start S3",
        "",
    ]

    for key in SCENARIO_SPECS:
        if key not in exported:
            lines.append(f"## {SCENARIO_SPECS[key]['title']}")
            lines.append("")
            lines.append("_No results recorded._")
            lines.append("")
            continue

        info = exported[key]
        spec = info["spec"]
        sub = info["df"]
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append(f"- CSV: `by_scenario/{os.path.basename(info['path'])}`")
        lines.append(f"- Eval setting: `{spec['eval_setting']}`")
        lines.append("")

        pivot_cols = ["config"]
        if "train_val_scenario" in sub.columns and sub["train_val_scenario"].nunique() > 1:
            pivot_cols = ["train_val_scenario", "config"]

        lines.append("| " + " | ".join(pivot_cols + ["macro_f1", "accuracy"]) + " |")
        lines.append("|" + "|".join(["---"] * (len(pivot_cols) + 2)) + "|")

        for keys, grp in sub.groupby(pivot_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = [str(k) for k in keys]
            f1 = grp["macro_f1"].astype(float).mean()
            acc = grp["accuracy"].astype(float).mean()
            lines.append("| " + " | ".join(labels + [f"{f1:.4f}", f"{acc:.4f}"]) + " |")

        lines.append("")

        if "config" in sub.columns and "full" in sub["config"].values:
            full_f1 = sub.loc[sub["config"] == "full", "macro_f1"].astype(float)
            for ab in ["wocon", "womv", "wocoattn"]:
                ab_f1 = sub.loc[sub["config"] == ab, "macro_f1"].astype(float)
                if len(full_f1) and len(ab_f1):
                    delta = float(full_f1.mean()) - float(ab_f1.mean())
                    lines.append(f"- Δmacro_f1 (full − {ab}): **{delta:+.4f}**")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Visualization (case study) is **not** included; run `case_study_vis.py` separately.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split all_results.csv by scenario")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = normalize_df(df)

    master_path = os.path.join(args.output_dir, "all_results.csv")
    df.to_csv(master_path, index=False)

    exported = export_by_scenario(df, args.output_dir)
    md = build_summary_markdown(df, exported, args.output_dir)
    report_path = os.path.join(args.output_dir, "REPORT_BY_SCENARIO.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[Export] master -> {master_path}")
    for key, info in exported.items():
        print(f"[Export] {SCENARIO_SPECS[key]['title']} -> {info['path']}")
    print(f"[Export] report -> {report_path}")


if __name__ == "__main__":
    main()
