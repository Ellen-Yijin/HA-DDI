# 实验脚本入口

本目录提供两套流水线，按需求选择：

| 场景 | 文档 | 命令 |
|------|------|------|
| **一次性全消融（1 seed，三场景）** | [README_SINGLE_RUN.md](README_SINGLE_RUN.md) | `python run_single_experiment.py` |
| **快速冒烟**（仅 1 config） | [README_SINGLE_RUN.md](README_SINGLE_RUN.md) | `python run_single_experiment.py --config full` |
| **论文终稿**（5 seeds + mean±std + 显著性） | 本目录 `run_multiseed_experiment.py` | `python run_multiseed_experiment.py --task both` |
| **Cold-start S2/S3 专项消融**（对比学习 / 多视图） | [README_COLDSTART_ABLATION.md](README_COLDSTART_ABLATION.md) | `python run_coldstart_ablation.py --fresh` |

## 一键命令

```bash
conda activate metddi
cd code/Classification/experiments

# 一次性全消融（默认：4 configs × 2 任务 × 3 评估场景）
python run_single_experiment.py

# 仅冒烟 1 个 config
python run_single_experiment.py --config full

# 正式多种子实验
python run_multiseed_experiment.py --task both --fresh

# Cold-start：S2/S3 上 full vs wocon vs womv（UnseenDrugs）
python run_coldstart_ablation.py --fresh
```

Windows：`run_single.bat` / `run_multiseed.bat`

## 文件一览

```
experiments/
├── README.md                  ← 本文件（入口索引）
├── README_SINGLE_RUN.md       ← 单次流程说明
├── README_MULTISEED.md        ← 多种子 + 统计检验说明
├── run_single_experiment.py   ← 默认入口：全消融 + 三场景
├── export_results_by_scenario.py
├── run_multiseed_experiment.py
├── aggregate_statistics.py
├── seed_utils.py
├── run_single.bat
├── run_multiseed.bat
├── run_coldstart_ablation.py
├── aggregate_coldstart_ablation.py
├── scenario_utils.py
├── README_COLDSTART_ABLATION.md
└── run_coldstart_ablation.bat
```
