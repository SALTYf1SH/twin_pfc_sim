@echo off
title Stress Model Ablation Training Pipeline
chcp 65001

:: =========================================================
:: CONFIGURATION
:: =========================================================

:: [关键] 请确认您的 Python 解释器路径
:: 根据您之前的报错信息，您的路径似乎是: E:/Anaconda/python.exe
set PYTHON_EXE=E:/Anaconda/python.exe

:: 训练脚本名称
set TRAIN_SCRIPT=train_stress_physics_ablation_fixed.py

:: 检查 Python 是否存在
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: %PYTHON_EXE%
    echo Please edit the .bat file and update PYTHON_EXE path.
    pause
    exit /b
)

:: 检查训练脚本是否存在
if not exist "%TRAIN_SCRIPT%" (
    echo [ERROR] Training script not found: %TRAIN_SCRIPT%
    echo Please make sure you are running this .bat inside 'new_dl_model' folder.
    pause
    exit /b
)

echo ========================================================
echo      STARTING BATCH ABLATION EXPERIMENTS
echo      Results will be logged to: trained_models_stress_ablation/experiment_log.csv
echo ========================================================
echo.

:: =========================================================
:: GROUP 1: Architecture Ablation (架构消融)
:: =========================================================

echo [1/8] Training: Full Dual-Branch Model (The Best Model)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [2/8] Training: Static-Only Architecture...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode static_only
if %errorlevel% neq 0 goto error

echo.
echo [3/8] Training: Dynamic-Only Architecture...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode dynamic_only
if %errorlevel% neq 0 goto error


:: =========================================================
:: GROUP 2: Mechanism/Loss Ablation (机理消融)
:: =========================================================

echo.
echo [4/8] Training: Baseline Model (MSE Only, No Physics)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation baseline --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [5/8] Training: No Topology Prior (No SSIM)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_ssim --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [6/8] Training: No Moving Arch Prior (No Arch)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_arch --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [7/8] Training: No Evolution Consistency (No Evo)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_evo --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [8/8] Training: No Continuity Prior (No TV)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_tv --branch_mode dual
if %errorlevel% neq 0 goto error


:: =========================================================
:: FINISH
:: =========================================================

echo.
echo ========================================================
echo      ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ========================================================
pause
exit /b

:error
echo.
echo [FATAL ERROR] An experiment failed. Batch process stopped.
pause
exit /b