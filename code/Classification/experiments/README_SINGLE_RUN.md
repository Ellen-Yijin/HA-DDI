# 一次性完整消融（训练 + 评估）

**一条命令**跑完：`full` + 全部消融（`wocon` / `womv` / `wocoattn`），覆盖 **三种评估场景**（transductive + cold-start S2/S3）。  
可视化（`case_study_vis.py`）**不在此脚本内**，请实验结束后再单独跑。

---

## 一键运行

```bash
conda activate metddi
cd code/Classification/experiments

python run_single_experiment.py
```

Windows:

```bat
run_single.bat
```

---

## 默认会跑什么

| 阶段 | 任务 | 训练 | 评估场景 |
|------|------|------|----------|
| 1 | **UnseenDDIs** | 4 configs × 1 seed | `test`（transductive） |
| 2 | **UnseenDrugs** | 4 configs，验证集 **S2** | `unseenone`（S2）+ `unseentwo`（S3） |

共 **8 次训练** + **12 条评估记录**（4 + 4×2）。

可选：加 `--include_s3_training` 再训一遍 UnseenDrugs（验证集 S3），用于最严格 cold-start 选模对比。

---

## 输出目录结构

```
experiments/results/full_ablation_YYYYMMDD_HHMMSS/
├── run_info.txt
├── all_results.csv                    # 全部指标汇总
├── REPORT_BY_SCENARIO.md              # 分场景摘要 + full vs 消融 Δ
├── by_scenario/
│   ├── 01_UnseenDDIs_transductive_test.csv
│   ├── 02_UnseenDrugs_coldstart_S2_unseenone.csv
│   └── 03_UnseenDrugs_coldstart_S3_unseentwo.csv
├── UnseenDDIs/
│   ├── weights_full_stage2.h5
│   ├── weights_wocon_stage2.h5
│   └── ...
└── UnseenDrugs/
    └── valS2/
        ├── weights_full_stage2.h5
        └── ...
```

`all_results.csv` 主要列：

| 列 | 含义 |
|----|------|
| `task` | UnseenDDIs / UnseenDrugs |
| `config` | full / wocon / womv / wocoattn |
| `dataset` | test / unseenone / unseentwo |
| `scenario` | test / S2 / S3 |
| `eval_setting` | transductive / coldstart |
| `train_val_scenario` | UnseenDrugs 训练时用的验证划分（S2 或 S3） |

---

## 常用参数

```bash
# 默认：4 配置 + seed 42 + 三场景评估
python run_single_experiment.py

# 指定种子 / 输出目录
python run_single_experiment.py --seed 42 --results_dir results/my_run

# 只评估已有权重
python run_single_experiment.py --skip_train --results_dir results/my_run

# 额外 UnseenDrugs S3 验证选模训练（+4 次训练）
python run_single_experiment.py --include_s3_training

# 只跑部分 config
python run_single_experiment.py --config full,wocon
```

---

## 与其他脚本的关系

| 脚本 | 用途 |
|------|------|
| **`run_single_experiment.py`** | **本次默认入口**：1 seed，全消融，三场景 |
| `run_multiseed_experiment.py` | 论文终稿：5 seeds + 显著性 |
| `run_coldstart_ablation.py` | 仅 UnseenDrugs S2/S3，多种子 cold-start 专项 |
| `case_study_vis.py` | 可视化（实验后单独跑） |

---

## 推荐工作流

```
1. python run_single_experiment.py          # 一次性全消融 + 三场景指标
2. 查看 REPORT_BY_SCENARIO.md / by_scenario/*.csv
3. （可选）run_multiseed_experiment.py      # 论文多种子
4. （可选）case_study_vis.py                # 可解释性可视化
```
