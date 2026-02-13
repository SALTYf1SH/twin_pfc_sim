@echo off
title Batch Training Stress Baselines

:: 1. 强制切换到脚本所在目录
cd /d "%~dp0"

:: 2. [关键修复] 指定您的 Conda 环境 Python 路径
:: 根据您之前的日志，您的路径是 E:\Anaconda\envs\mamba\python.exe
:: 如果路径不对，请修改下面这一行
set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe

echo ========================================================
echo Current Directory: %CD%
echo Using Python: %PYTHON_EXE%
echo ========================================================
echo.

:: 3. 检查 Python 是否存在
if not exist "%PYTHON_EXE%" (
    echo [CRITICAL ERROR]
    echo Python executable not found at: %PYTHON_EXE%
    echo Please check the path in the .bat file.
    pause
    exit /b 1
)

:: 4. Train Deep CNN
echo [1/3] Training Deep CNN Baseline...
"%PYTHON_EXE%" train_baselines_stress_mamba_fixed.py --model_type CNN
if %errorlevel% neq 0 (
    echo [ERROR] CNN Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] CNN Finished.
echo.

:: 5. Train Bi-LSTM
echo [2/3] Training Bi-LSTM Baseline...
"%PYTHON_EXE%" train_baselines_stress_mamba_fixed.py --model_type LSTM
if %errorlevel% neq 0 (
    echo [ERROR] LSTM Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] LSTM Finished.
echo.

:: 6. Train Mamba
echo [3/3] Training Mamba Baseline...
"%PYTHON_EXE%" train_baselines_stress_mamba_fixed.py --model_type MAMBA
if %errorlevel% neq 0 (
    echo [ERROR] Mamba Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Mamba Finished.
echo.

echo ========================================================
echo      ALL BASELINE MODELS TRAINED SUCCESSFULLY
echo ========================================================
pause