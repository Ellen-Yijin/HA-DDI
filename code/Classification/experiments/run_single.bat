@echo off
REM 一次性完整消融：UnseenDDIs(transductive) + UnseenDrugs(S2/S3 cold-start)
REM 配置：full / wocon / womv / wocoattn
REM 使用前: conda activate metddi

cd /d "%~dp0"
python run_single_experiment.py %*

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Full ablation run failed.
    exit /b %ERRORLEVEL%
)

echo [OK] Done. See experiments\results\full_ablation_* for outputs.
