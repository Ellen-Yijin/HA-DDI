@echo off
REM Cold-start S2/S3 专项消融（对比学习 / 多视图）
REM 使用前: conda activate metddi

cd /d "%~dp0"
python run_coldstart_ablation.py %*
