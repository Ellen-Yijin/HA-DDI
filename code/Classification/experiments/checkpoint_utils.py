"""Checkpoint / resume helpers for ablation training pipelines."""
from __future__ import annotations

import csv
import os
from typing import Iterable


def weights_stage1_path(weights_dir: str, config: str) -> str:
    return os.path.join(weights_dir, f"weights_{config}_stage1.weights.h5")


def weights_stage2_path(weights_dir: str, config: str) -> str:
    return os.path.join(weights_dir, f"weights_{config}_stage2.weights.h5")


def training_status(weights_dir: str, config: str) -> str:
    """Return 'complete' | 'stage1_only' | 'missing'."""
    s2 = weights_stage2_path(weights_dir, config)
    s1 = weights_stage1_path(weights_dir, config)
    if os.path.isfile(s2):
        return "complete"
    if os.path.isfile(s1):
        return "stage1_only"
    return "missing"


def should_skip_training(
    weights_dir: str,
    config: str,
    *,
    resume: bool,
    force_train: bool,
) -> bool:
    if force_train or not resume:
        return False
    return training_status(weights_dir, config) == "complete"


def _read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_matches(
    row: dict[str, str],
    *,
    task: str,
    seed: int,
    config: str,
    ablation_suite: str,
    dataset: str | None = None,
    scenario: str | None = None,
    train_val_scenario: str | None = None,
) -> bool:
    if row.get("task") != task:
        return False
    if str(row.get("seed")) != str(seed):
        return False
    if row.get("config") != config:
        return False
    if row.get("ablation_suite") != ablation_suite:
        return False
    if dataset is not None and row.get("dataset") != dataset:
        return False
    if scenario is not None and row.get("scenario") != scenario:
        return False
    if train_val_scenario is not None and row.get("train_val_scenario") != train_val_scenario:
        return False
    return True


def unseen_ddis_eval_done(
    csv_path: str,
    seed: int,
    config: str,
    ablation_suite: str,
) -> bool:
    for row in _read_csv_rows(csv_path):
        if _row_matches(
            row,
            task="UnseenDDIs",
            seed=seed,
            config=config,
            ablation_suite=ablation_suite,
            dataset="test",
            scenario="test",
        ):
            return True
    return False


def unseen_drugs_eval_done(
    csv_path: str,
    seed: int,
    config: str,
    ablation_suite: str,
    train_val_scenario: str,
    eval_scenarios: Iterable[str],
) -> bool:
    scenario_dataset = {"S2": "unseenone", "S3": "unseentwo"}
    rows = _read_csv_rows(csv_path)
    for sc in eval_scenarios:
        ds = scenario_dataset.get(sc, sc)
        found = False
        for row in rows:
            if _row_matches(
                row,
                task="UnseenDrugs",
                seed=seed,
                config=config,
                ablation_suite=ablation_suite,
                dataset=ds,
                scenario=sc,
                train_val_scenario=train_val_scenario,
            ):
                found = True
                break
        if not found:
            return False
    return True


def print_training_status(weights_dir: str, configs: list[str]) -> None:
    print(f"[Resume] checkpoint status under {weights_dir}:")
    for cfg in configs:
        status = training_status(weights_dir, cfg)
        print(f"  {cfg}: {status}")
