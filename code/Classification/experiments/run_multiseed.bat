@echo off
REM 多种子实验一键启动（Windows）
REM 使用前请先激活 conda 环境，例如: conda activate metddi

cd /d "%~dp0"

python run_multiseed_experiment.py --task both %*

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Experiment failed.
    exit /b %ERRORLEVEL%
)

echo [OK] Experiment finished. Check experiments\results\ for outputs.
