@echo off
title Subsidence Model Ablation Training Pipeline
chcp 65001

:: =========================================================
:: CONFIGURATION
:: =========================================================

:: [关键] 请确认 Python 路径
set PYTHON_EXE=E:/Anaconda/python.exe

:: 训练脚本名称
set TRAIN_SCRIPT=train_subsidence_physics_ablation_fixed.py

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: %PYTHON_EXE%
    pause
    exit /b
)

if not exist "%TRAIN_SCRIPT%" (
    echo [ERROR] Script not found: %TRAIN_SCRIPT%
    pause
    exit /b
)

echo ========================================================
echo      STARTING SUBSIDENCE ABLATION EXPERIMENTS
echo      Results: trained_models_subsidence_ablation/experiment_log.csv
echo ========================================================
echo.

:: --- Architecture Ablation ---
echo [1/8] Full Model (Dual Branch)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [2/8] Static Only...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode static_only
if %errorlevel% neq 0 goto error

echo.
echo [3/8] Dynamic Only...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation full --branch_mode dynamic_only
if %errorlevel% neq 0 goto error

:: --- Loss Ablation ---
echo.
echo [4/8] Baseline (MSE Only)...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation baseline --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [5/8] No SSIM...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_ssim --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [6/8] No Arch Prior...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_arch --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [7/8] No Evo Prior...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_evo --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo [8/8] No TV Prior...
"%PYTHON_EXE%" %TRAIN_SCRIPT% --ablation no_tv --branch_mode dual
if %errorlevel% neq 0 goto error

echo.
echo ========================================================
echo      ALL SUBSIDENCE EXPERIMENTS COMPLETED!
echo ========================================================
pause
exit /b

:error
echo.
echo [FATAL ERROR] Experiment failed.
pause
exit /b