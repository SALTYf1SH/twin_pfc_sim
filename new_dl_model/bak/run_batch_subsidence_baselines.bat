@echo off
title Batch Training Subsidence Baselines

:: 1. 强制切换到脚本所在目录 (解决路径找不到问题)
cd /d "%~dp0"

:: 2. [关键] 指定您的 Anaconda 环境 Python 路径 (解决 Numpy 找不到问题)
:: 根据您之前的日志，您的路径是 E:\Anaconda\envs\mamba\python.exe
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

:: 4. 检查训练脚本是否存在
if not exist "train_baselines_subsidence_mamba_fixed.py" (
    echo [CRITICAL ERROR]
    echo Script 'train_baselines_subsidence_mamba_fixed.py' not found!
    pause
    exit /b 1
)

:: ----------------------------------------------------------
:: 开始训练流程
:: ----------------------------------------------------------

:: 1. Train Deep CNN
echo [1/3] Training Deep CNN Baseline...
"%PYTHON_EXE%" train_baselines_subsidence_mamba_fixed.py --model_type CNN
if %errorlevel% neq 0 (
    echo [ERROR] CNN Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] CNN Finished.
echo.

:: 2. Train Bi-LSTM
echo [2/3] Training Bi-LSTM Baseline...
"%PYTHON_EXE%" train_baselines_subsidence_mamba_fixed.py --model_type LSTM
if %errorlevel% neq 0 (
    echo [ERROR] LSTM Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] LSTM Finished.
echo.

:: 3. Train Mamba
echo [3/3] Training Mamba Baseline...
"%PYTHON_EXE%" train_baselines_subsidence_mamba_fixed.py --model_type MAMBA
if %errorlevel% neq 0 (
    echo [ERROR] Mamba Training Failed!
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Mamba Finished.
echo.

echo ========================================================
echo      ALL SUBSIDENCE BASELINES TRAINED SUCCESSFULLY
echo ========================================================
pause