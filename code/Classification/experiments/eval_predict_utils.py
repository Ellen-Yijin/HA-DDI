"""Shared evaluation predict helpers (eager mode + full-dataset predict)."""
from __future__ import annotations

import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

EVAL_CSV_HEADER = [
    "task",
    "seed",
    "config",
    "dataset",
    "scenario",
    "eval_setting",
    "ablation_suite",
    "train_val_scenario",
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_aupr",
]


def configure_eval_runtime() -> None:
    """Match training: eager execution, disable XLA JIT to avoid long compile stalls."""
    tf.config.run_functions_eagerly(True)
    tf.config.optimizer.set_jit(False)
    print(
        "[Eval] runtime: run_functions_eagerly=True, XLA JIT=False (same as training)",
        flush=True,
    )


def compute_macro_aupr(y_true, y_prob) -> float:
    """Macro-averaged AUPR (one-vs-rest) for multi-class softmax outputs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    classes = list(range(n_classes))
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return float(average_precision_score(y_true_bin, y_prob, average="macro"))


def ensure_eval_csv_header(csv_path: str) -> bool:
    """
    Ensure CSV header includes macro_aupr.
    Returns True if the file already existed.
    """
    if not os.path.isfile(csv_path):
        return False

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return True

    header = rows[0]
    if "macro_aupr" in header:
        return True

    header = header + ["macro_aupr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows[1:]:
            writer.writerow(row + [""])

    print(f"[Eval] upgraded CSV header with macro_aupr: {csv_path}", flush=True)
    return True


def eval_one_dataset(model, dataset, tag: str):
    """
    Predict on the full dataset in one model.predict() call with Keras progress bar.
    Labels are collected once via .cache() so the data pipeline is not duplicated.
    """
    print(f"\n[Eval-{tag}] caching dataset...", flush=True)
    cached = dataset.cache()

    print(f"[Eval-{tag}] collecting labels...", flush=True)
    y_parts = []
    n_batches = 0
    for _, batch_y in cached:
        y_parts.append(batch_y.numpy().ravel())
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"[Eval-{tag}] labels cached: {n_batches} batches", flush=True)

    y_true = np.concatenate(y_parts)
    print(f"[Eval-{tag}] {len(y_true)} samples, starting full-dataset predict...", flush=True)

    y_prob = model.predict(cached.map(lambda x, y: x), verbose=1)
    y_pred = np.argmax(y_prob, axis=-1).ravel()

    print(f"\n========== [{tag}] classification_report ==========", flush=True)
    print(classification_report(y_true, y_pred))
    print(f"========== [{tag}] confusion_matrix ==========")
    print(confusion_matrix(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_aupr = compute_macro_aupr(y_true, y_prob)

    print(
        f"[Eval-{tag}] accuracy={acc:.4f}, "
        f"macro_f1={macro_f1:.4f}, "
        f"macro_precision={macro_precision:.4f}, "
        f"macro_recall={macro_recall:.4f}, "
        f"macro_aupr={macro_aupr:.4f}",
        flush=True,
    )

    return acc, macro_f1, macro_precision, macro_recall, macro_aupr
