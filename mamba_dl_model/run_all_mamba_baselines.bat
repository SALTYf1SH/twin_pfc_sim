@echo off
echo =================================================================
echo  Running ALL Baseline Training - Stress and Subsidence - Mamba Comparison
echo =================================================================

set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe

echo.
echo [1/6] Training Stress CNN Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_stress.py" --model_type CNN

echo.
echo [2/6] Training Stress LSTM Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_stress.py" --model_type LSTM

echo.
echo [3/6] Training Stress Transformer Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_stress.py" --model_type TRANSFORMER

echo.
echo [4/6] Training Subsidence CNN Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_subsidence.py" --model_type CNN

echo.
echo [5/6] Training Subsidence LSTM Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_subsidence.py" --model_type LSTM

echo.
echo [6/6] Training Subsidence Transformer Baseline...
"%PYTHON_EXE%" "%~dp0train_baselines_subsidence.py" --model_type TRANSFORMER

echo.
echo =================================================================
echo  Baseline Training Completed!
echo =================================================================
pause
