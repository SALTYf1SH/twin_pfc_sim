@echo off
echo =================================================================
echo  Running Robustness Batch Evaluation
echo =================================================================

set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe

echo.
echo [1/2] Evaluating Mamba Ablation Models...
"%PYTHON_EXE%" "%~dp0scripts_eval\batch_evaluate_robustness_mamba.py"

echo.
echo [2/2] Evaluating Baseline Models...
"%PYTHON_EXE%" "%~dp0scripts_eval\batch_evaluate_robustness_baselines.py"

echo.
echo =================================================================
echo  Evaluation Completed!
echo =================================================================
pause
