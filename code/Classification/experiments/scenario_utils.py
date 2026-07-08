"""Inductive / cold-start 场景常量（S2、S3）与数据集映射。"""
from __future__ import annotations

# S1: 两药均在训练集见过（transductive）
# S2: 仅一药未见过（unseen one drug）
# S3: 两药均未见过（unseen two drugs）
SCENARIO_S2 = "S2"
SCENARIO_S3 = "S3"

DATASET_TO_SCENARIO = {
    "unseenone": SCENARIO_S2,
    "unseentwo": SCENARIO_S3,
    # 兼容别名
    "S2": SCENARIO_S2,
    "S3": SCENARIO_S3,
}

VAL_SCENARIO_CHOICES = (SCENARIO_S2, SCENARIO_S3)

# cold-start 消融默认对比：完整模型 vs 去对比学习 vs 去多视图
ALL_ABLATION_CONFIGS = ["full", "wocon", "womv", "wocoattn"]
COLDSTART_DEFAULT_CONFIGS = ["full", "wocon", "womv"]
COLDSTART_CONTRASTIVE_COMPARISONS = ["wocon"]
COLDSTART_MULTIVIEW_COMPARISONS = ["womv"]


def dataset_tag_to_scenario(dataset_tag: str) -> str:
    return DATASET_TO_SCENARIO.get(dataset_tag, dataset_tag)


def parse_val_scenario(name: str) -> str:
    key = name.strip()
    if key not in VAL_SCENARIO_CHOICES:
        raise ValueError(f"val_scenario must be one of {VAL_SCENARIO_CHOICES}, got {name!r}")
    return key
