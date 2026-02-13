@echo off
echo ========================================================
echo   Starting Parallel Robustness Training (Mamba + Baselines)
echo   Seeds: 10, 15, 29, 53, 65, 77, 88, 92
echo ========================================================

set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe
set SCRIPT_DIR=%~dp0mamba_dl_model\scripts_train

if not exist "mamba_dl_model\robustness_results_stress" mkdir "mamba_dl_model\robustness_results_stress"
if not exist "mamba_dl_model\robustness_results_subsidence" mkdir "mamba_dl_model\robustness_results_subsidence"
if not exist "mamba_dl_model\robustness_results_stress_baselines" mkdir "mamba_dl_model\robustness_results_stress_baselines"
if not exist "mamba_dl_model\robustness_results_subsidence_baselines" mkdir "mamba_dl_model\robustness_results_subsidence_baselines"

set SEEDS=10 15 29 53 65 77 88 92

:: Launch one window per seed. Each window processes Mamba + Baselines sequentially.
for %%s in (%SEEDS%) do (
    echo Launching training for SEED %%s...
    start "Train_Seed_%%s" cmd /c "echo Running ALL Models for Seed %%s... && ^
    echo [1/8] Stress: Mamba Full Dual... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %%s --ablation_name full --branch_mode dual && ^
    echo [2/8] Stress: Mamba Dynamic Only... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %%s --ablation_name full --branch_mode dynamic_only && ^
    echo [3/8] Stress: Mamba Static Only... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %%s --ablation_name full --branch_mode static_only && ^
    echo [4/8] Stress: Mamba No Physics... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_stress_robustness.py --seed %%s --ablation_name no_physics --branch_mode dual --no_physics && ^
    echo [5/8] Stress: Baselines (MAMBA, CNN, LSTM, TRANSFORMER)... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type MAMBA --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type CNN --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type LSTM --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_stress_robustness.py --model_type TRANSFORMER --seed %%s && ^
    echo [6/8] Subsidence: Mamba Full Dual... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %%s --ablation_name full --branch_mode dual && ^
    echo [7/8] Subsidence: Mamba Ablations... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %%s --ablation_name full --branch_mode dynamic_only && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %%s --ablation_name full --branch_mode static_only && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_subsidence_robustness.py --seed %%s --ablation_name no_physics --branch_mode dual --no_physics && ^
    echo [8/8] Subsidence: Baselines... && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type MAMBA --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type CNN --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type LSTM --seed %%s && ^
    %PYTHON_EXE% %SCRIPT_DIR%\train_baselines_subsidence_robustness.py --model_type TRANSFORMER --seed %%s && ^
    echo Done with Seed %%s! && pause"
)

echo.
echo All training workers dispatched.
pause
