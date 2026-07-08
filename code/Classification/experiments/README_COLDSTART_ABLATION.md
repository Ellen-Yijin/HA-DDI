# Cold-Start 专项消融（S2 / S3）

在 **UnseenDrugs** inductive 设定下，对比 `full` vs `wocon`（去 Motif-Dropping 对比学习）vs `womv`（去多视图融合）。  
Transductive 测试上掉点往往很小；**S2/S3 上对比分支的贡献通常更明显**，更适合写 cold-start 鲁棒性。

## 场景定义

| 代号 | 数据集文件 | 含义 |
|------|-----------|------|
| **S2** | `val_dataset_unseen_onedrug.csv` | 药物对中 **1 个药物** 未在训练集出现 |
| **S3** | `val_dataset_unseen_twodrugs.csv` | **两个药物** 均未在训练集出现 |

> UnseenDDIs 的 `tst_dataset` 几乎全为 S1（两药均见过），**不适合** S2/S3 cold-start 消融；本流水线针对 UnseenDrugs。

## 一键运行

```bash
conda activate metddi
cd code/Classification/experiments

# 默认：5 seeds × (full,wocon,womv) × 训练 val=S2 与 val=S3，评估 S2+S3
python run_coldstart_ablation.py --fresh

# 仅 S2 训练验证（更快，与旧脚本行为一致）
python run_coldstart_ablation.py --train_val_scenarios S2 --configs full,wocon

# 只汇总已有结果
python run_coldstart_ablation.py --aggregate_only --results_dir results/coldstart_YYYYMMDD_HHMMSS
```

Windows: `run_coldstart_ablation.bat --fresh`

## 输出

```
results/coldstart_*/
├── raw_runs.csv                 # 含 scenario / train_val_scenario / ablation_suite 列
├── coldstart_summary.csv        # 按 train_val × eval × config 的 mean±std
├── coldstart_significance.csv   # full vs wocon / womv 配对检验
├── REPORT_COLDSTART.md          # 论文用摘要表
└── UnseenDrugs/
    ├── valS2/seed_42/weights_full_stage2.h5
    └── valS3/seed_42/weights_full_stage2.h5
```

## 训练 / 评估参数

**训练**（`Train_UnseenDrugs.py`）新增：

```bash
--val_scenario S2   # 默认；early stopping 用 unseen one drug
--val_scenario S3   # early stopping 用 unseen two drugs（更严格 cold-start 选模）
```

**评估**（`UnseenDrugsnew_evaluate.py`）新增：

```bash
--train_val_scenario S2      # 记录训练时用的 val 划分
--eval_scenarios S2,S3       # 在哪些 inductive 集上评估
--ablation_suite coldstart_contrastive
```

## 论文怎么写

1. **主表**：`train_val_scenario=S2` 时，S2 与 S3 上的 full / wocon / womv macro-F1（mean±std）。
2. **对比学习论点**：引用 `full vs wocon` 在 **S3** 上的 Δmacro_f1 与 Wilcoxon p 值。
3. **选模说明**：若使用 `train_val_scenario=S3` 训练的权重，在正文说明「checkpoint 按最严格 cold-start 验证集选取」。

## 与通用消融的关系

| 流水线 | 用途 |
|--------|------|
| `run_multiseed_experiment.py` | 全配置 + UnseenDDIs/Drugs 标准评估 |
| `run_coldstart_ablation.py` | **仅 UnseenDrugs S2/S3**，聚焦 contrastive / multi-view |

两者 CSV 格式兼容；cold-start 结果带 `scenario` 与 `ablation_suite=coldstart_contrastive` 列。
