#!/usr/bin/env python3
"""
汇总多种子实验 raw_runs.csv：
  - 计算各指标 mean ± std
  - 对 full vs 各消融配置做配对 Wilcoxon 符号秩检验与配对 t 检验
  - 输出 CSV 与 Markdown 报告
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_aupr"]
BASELINE = "full"
COMPARISONS = ["wocon", "womv", "wocoattn"]


def significance_stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
            "ttest_stat": np.nan,
            "ttest_p": np.nan,
        }

    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    t_stat, t_p = stats.ttest_rel(a, b)
    return {
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容旧版 CSV（无 task/seed/scenario 列）。"""
    df = df.copy()
    if "task" not in df.columns:
        df["task"] = "Unknown"
    if "seed" not in df.columns:
        df["seed"] = range(len(df))
    if "scenario" not in df.columns and "dataset" in df.columns:
        _map = {"unseenone": "S2", "unseentwo": "S3", "test": "test"}
        df["scenario"] = df["dataset"].map(lambda x: _map.get(x, x))
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["task", "dataset", "config"]
    for keys, grp in df.groupby(group_cols):
        task, dataset, config = keys
        for metric in METRICS:
            vals = grp[metric].astype(float).values
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "config": config,
                    "metric": metric,
                    "n_runs": len(vals),
                    "mean": np.mean(vals),
                    "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                    "mean_pm_std": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}",
                }
            )
    return pd.DataFrame(rows)


def build_significance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, dataset), grp in df.groupby(["task", "dataset"]):
        for metric in METRICS:
            pivot = grp.pivot_table(index="seed", columns="config", values=metric, aggfunc="first")
            if BASELINE not in pivot.columns:
                continue
            baseline = pivot[BASELINE].dropna()
            for comp in COMPARISONS:
                if comp not in pivot.columns:
                    continue
                common_seeds = baseline.index.intersection(pivot[comp].dropna().index)
                if len(common_seeds) < 2:
                    continue
                a = pivot.loc[common_seeds, BASELINE].astype(float).values
                b = pivot.loc[common_seeds, comp].astype(float).values
                tests = paired_tests(a, b)
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": BASELINE,
                        "comparison": comp,
                        "n_pairs": len(common_seeds),
                        "baseline_mean": np.mean(a),
                        "comparison_mean": np.mean(b),
                        "mean_diff": np.mean(a) - np.mean(b),
                        **tests,
                        "wilcoxon_sig": significance_stars(tests["wilcoxon_p"]),
                        "ttest_sig": significance_stars(tests["ttest_p"]),
                    }
                )
    return pd.DataFrame(rows)


def build_markdown(summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    lines = ["# Multi-Seed Experiment Report", ""]
    lines.append("## Summary (mean ± std)")
    lines.append("")

    for task in summary["task"].unique():
        lines.append(f"### {task}")
        lines.append("")
        sub = summary[summary["task"] == task]
        for dataset in sub["dataset"].unique():
            lines.append(f"#### Dataset: `{dataset}`")
            lines.append("")
            lines.append("| Config | Metric | Result | n |")
            lines.append("|--------|--------|--------|---|")
            ds_sub = sub[sub["dataset"] == dataset]
            for _, row in ds_sub.iterrows():
                lines.append(
                    f"| {row['config']} | {row['metric']} | {row['mean_pm_std']} | {int(row['n_runs'])} |"
                )
            lines.append("")

    lines.append("## Significance Tests (full vs ablations)")
    lines.append("")
    lines.append("Paired tests across seeds. Wilcoxon: non-parametric; t-test: parametric.")
    lines.append("Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001, ns = not significant")
    lines.append("")

    if significance.empty:
        lines.append("_No significance tests computed (insufficient paired runs)._")
        return "\n".join(lines)

    for task in significance["task"].unique():
        lines.append(f"### {task}")
        lines.append("")
        sub = significance[significance["task"] == task]
        lines.append(
            "| Dataset | Metric | Compare | Δmean | Wilcoxon p | Wilcoxon | t-test p | t-test | n |"
        )
        lines.append("|---------|--------|---------|-------|------------|----------|----------|--------|---|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['metric']} | {row['comparison']} "
                f"| {row['mean_diff']:+.4f} "
                f"| {row['wilcoxon_p']:.4g} | {row['wilcoxon_sig']} "
                f"| {row['ttest_p']:.4g} | {row['ttest_sig']} "
                f"| {int(row['n_pairs'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results")
    parser.add_argument("--input", type=str, required=True, help="Path to raw_runs.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for summary outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = normalize_columns(df)

    os.makedirs(args.output_dir, exist_ok=True)

    summary = build_summary(df)
    significance = build_significance(df)
    markdown = build_markdown(summary, significance)

    summary_path = os.path.join(args.output_dir, "summary_mean_std.csv")
    sig_path = os.path.join(args.output_dir, "significance_tests.csv")
    md_path = os.path.join(args.output_dir, "REPORT.md")

    summary.to_csv(summary_path, index=False)
    significance.to_csv(sig_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[Aggregate] summary -> {summary_path}")
    print(f"[Aggregate] significance -> {sig_path}")
    print(f"[Aggregate] report -> {md_path}")

    print("\n--- Quick preview (macro_f1) ---")
    preview = summary[summary["metric"] == "macro_f1"][["task", "dataset", "config", "mean_pm_std"]]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
