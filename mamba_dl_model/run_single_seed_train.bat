@echo off
:: Access Seed from argument
set SEED=%1
if "%SEED%"=="" (
    echo Error: No seed provided.
    pause
    exit /b
)

:: Switch to Project Root (Parent of mamba_dl_model)
cd /d %~dp0..
set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe
:: Script dir is inside mamba_dl_model/scripts_train
set SCRIPT_DIR=mamba_dl_model\scripts_train

echo ===========================================
echo Running Training Sequence for SEED %SEED%
echo ===========================================

echo [1/8] Stress: Mamba Full Dual...
%PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %SEED% --ablation_name full --branch_mode dual

echo [2/8] Stress: Mamba Dynamic Only...
%PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %SEED% --ablation_name full --branch_mode dynamic_only

echo [3/8] Stress: Mamba Static Only...
%PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %SEED% --ablation_name full --branch_mode static_only

echo [4/8] Stress: Mamba No Physics...
%PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %SEED% --ablation_name no_physics --branch_mode dual --no_physics

echo [5/8] Stress: Baselines (MAMBA, CNN, LSTM, TRANSFORMER)...
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type MAMBA --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type CNN --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type LSTM --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type TRANSFORMER --seed %SEED%

echo [6/8] Subsidence: Mamba Full Dual...
%PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %SEED% --ablation_name full --branch_mode dual

echo [7/8] Subsidence: Mamba Ablations...
%PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %SEED% --ablation_name full --branch_mode dynamic_only
%PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %SEED% --ablation_name full --branch_mode static_only
%PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %SEED% --ablation_name no_physics --branch_mode dual --no_physics

echo [8/8] Subsidence: Baselines...
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type MAMBA --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type CNN --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type LSTM --seed %SEED%
%PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type TRANSFORMER --seed %SEED%

echo.
echo ===========================================
echo FINISHED SEED %SEED%
echo ===========================================
pause
exit
