#!/usr/bin/env python3
"""
汇总 cold-start（S2/S3）专项消融结果。

重点对比：
  - full vs wocon  → 对比学习（Motif Dropping）在 inductive 场景的贡献
  - full vs womv   → 多视图融合在 inductive 场景的贡献

分组维度：train_val_scenario × scenario(eval) × config
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from scenario_utils import (
    COLDSTART_CONTRASTIVE_COMPARISONS,
    COLDSTART_MULTIVIEW_COMPARISONS,
    DATASET_TO_SCENARIO,
)

METRICS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_aupr"]
BASELINE = "full"
COMPARISON_GROUPS = {
    "contrastive": COLDSTART_CONTRASTIVE_COMPARISONS,
    "multiview": COLDSTART_MULTIVIEW_COMPARISONS,
}


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
        return {"wilcoxon_p": np.nan, "ttest_p": np.nan}
    try:
        _, w_p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        w_p = np.nan
    _, t_p = stats.ttest_rel(a, b)
    return {"wilcoxon_p": w_p, "ttest_p": t_p}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "scenario" not in df.columns and "dataset" in df.columns:
        df["scenario"] = df["dataset"].map(lambda x: DATASET_TO_SCENARIO.get(x, x))
    if "train_val_scenario" not in df.columns:
        df["train_val_scenario"] = "S2"
    if "ablation_suite" not in df.columns:
        df["ablation_suite"] = "standard"
    if "seed" not in df.columns:
        df["seed"] = range(len(df))
    return df


def filter_coldstart(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_df(df)
    mask = df["scenario"].isin(["S2", "S3"])
    if "task" in df.columns:
        mask &= df["task"].astype(str).str.contains("UnseenDrugs", na=False)
    cold = df.loc[mask].copy()
    if cold.empty:
        cold = df.loc[df["scenario"].isin(["S2", "S3"])].copy()
    return cold


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["train_val_scenario", "scenario", "config"]
    for keys, grp in df.groupby(group_cols):
        train_val, scenario, config = keys
        for metric in METRICS:
            vals = grp[metric].astype(float).values
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            rows.append(
                {
                    "train_val_scenario": train_val,
                    "eval_scenario": scenario,
                    "config": config,
                    "metric": metric,
                    "n_runs": len(vals),
                    "mean": np.mean(vals),
                    "std": std,
                    "mean_pm_std": f"{np.mean(vals):.4f} ± {std:.4f}",
                }
            )
    return pd.DataFrame(rows)


def build_significance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (train_val, scenario), grp in df.groupby(["train_val_scenario", "scenario"]):
        for ablation_type, comparisons in COMPARISON_GROUPS.items():
            for metric in METRICS:
                pivot = grp.pivot_table(
                    index="seed", columns="config", values=metric, aggfunc="first"
                )
                if BASELINE not in pivot.columns:
                    continue
                for comp in comparisons:
                    if comp not in pivot.columns:
                        continue
                    common = pivot[BASELINE].dropna().index.intersection(pivot[comp].dropna().index)
                    if len(common) < 2:
                        continue
                    a = pivot.loc[common, BASELINE].astype(float).values
                    b = pivot.loc[common, comp].astype(float).values
                    tests = paired_tests(a, b)
                    rows.append(
                        {
                            "train_val_scenario": train_val,
                            "eval_scenario": scenario,
                            "ablation_type": ablation_type,
                            "metric": metric,
                            "baseline": BASELINE,
                            "comparison": comp,
                            "n_pairs": len(common),
                            "full_mean": np.mean(a),
                            "ablation_mean": np.mean(b),
                            "delta_full_minus_ablation": np.mean(a) - np.mean(b),
                            **tests,
                            "wilcoxon_sig": significance_stars(tests["wilcoxon_p"]),
                            "ttest_sig": significance_stars(tests["ttest_p"]),
                        }
                    )
    return pd.DataFrame(rows)


def build_markdown(summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    lines = [
        "# Cold-Start Ablation Report (S2 / S3)",
        "",
        "S2 = unseen one drug (`unseenone`); S3 = unseen two drugs (`unseentwo`).",
        "",
        "Δ = mean(full) − mean(ablation). **正值表示完整模型优于消融配置。**",
        "",
    ]

    if summary.empty:
        lines.append("_No results._")
        return "\n".join(lines)

    lines.append("## Results (mean ± std)")
    lines.append("")
    for train_val in sorted(summary["train_val_scenario"].unique()):
        lines.append(f"### Training validation: `{train_val}`")
        lines.append("")
        sub = summary[summary["train_val_scenario"] == train_val]
        for eval_sc in sorted(sub["eval_scenario"].unique()):
            lines.append(f"#### Eval scenario `{eval_sc}`")
            lines.append("")
            lines.append("| Config | macro_f1 | accuracy | n |")
            lines.append("|--------|----------|----------|---|")
            ds = sub[sub["eval_scenario"] == eval_sc]
            for config in sorted(ds["config"].unique()):
                f1_row = ds[(ds["config"] == config) & (ds["metric"] == "macro_f1")]
                acc_row = ds[(ds["config"] == config) & (ds["metric"] == "accuracy")]
                if f1_row.empty:
                    continue
                f1 = f1_row.iloc[0]["mean_pm_std"]
                acc = acc_row.iloc[0]["mean_pm_std"] if not acc_row.empty else "n/a"
                n = int(f1_row.iloc[0]["n_runs"])
                lines.append(f"| {config} | {f1} | {acc} | {n} |")
            lines.append("")

    lines.append("## Significance: full vs ablations (paired across seeds)")
    lines.append("")
    if significance.empty:
        lines.append("_Insufficient paired runs for significance tests._")
        return "\n".join(lines)

    focus = significance[significance["metric"] == "macro_f1"].copy()
    lines.append("| Train val | Eval | Ablation | Compare | Δ macro_f1 | Wilcoxon p | Sig | n |")
    lines.append("|-----------|------|----------|---------|------------|------------|-----|---|")
    for _, row in focus.sort_values(["train_val_scenario", "eval_scenario", "ablation_type"]).iterrows():
        lines.append(
            f"| {row['train_val_scenario']} | {row['eval_scenario']} | {row['ablation_type']} "
            f"| {row['comparison']} | {row['delta_full_minus_ablation']:+.4f} "
            f"| {row['wilcoxon_p']:.4g} | {row['wilcoxon_sig']} | {int(row['n_pairs'])} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**论文表述建议**：优先引用 S3（两药均未见）上 full vs wocon 的 Δmacro_f1；")
    lines.append("若 train_val=S3 的模型在 S3 上增益更大，可强调「针对最严格 cold-start 选模」。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cold-start S2/S3 ablation CSV")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = filter_coldstart(df)

    os.makedirs(args.output_dir, exist_ok=True)

    summary = build_summary(df)
    significance = build_significance(df)
    markdown = build_markdown(summary, significance)

    summary_path = os.path.join(args.output_dir, "coldstart_summary.csv")
    sig_path = os.path.join(args.output_dir, "coldstart_significance.csv")
    md_path = os.path.join(args.output_dir, "REPORT_COLDSTART.md")

    summary.to_csv(summary_path, index=False)
    significance.to_csv(sig_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[ColdStart] summary -> {summary_path}")
    print(f"[ColdStart] significance -> {sig_path}")
    print(f"[ColdStart] report -> {md_path}")

    print("\n--- macro_f1 on S2/S3 (full vs wocon) ---")
    if not significance.empty:
        preview = significance[
            (significance["metric"] == "macro_f1")
            & (significance["comparison"] == "wocon")
        ][["train_val_scenario", "eval_scenario", "delta_full_minus_ablation", "wilcoxon_sig", "n_pairs"]]
        print(preview.to_string(index=False))
    else:
        preview = summary[summary["metric"] == "macro_f1"]
        print(preview[["train_val_scenario", "eval_scenario", "config", "mean_pm_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
