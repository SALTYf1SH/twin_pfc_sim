@echo off
echo =================================================================
echo  Running Mamba Ablation Studies - Stress and Subsidence
echo =================================================================

set PYTHON_EXE=E:\Anaconda\envs\mamba\python.exe

REM --- Stress Ablations ---

echo.
echo [1/8] Stress - Full Physics (Mamba Dual)
"%PYTHON_EXE%" "%~dp0train_stress_physics_mamba.py" --ablation_name full --branch_mode dual

echo.
echo [2/8] Stress - No Physics Loss (Mamba Dual)
"%PYTHON_EXE%" "%~dp0train_stress_physics_mamba.py" --ablation_name no_physics --branch_mode dual --no_physics

echo.
echo [3/8] Stress - Static Branch Only
"%PYTHON_EXE%" "%~dp0train_stress_physics_mamba.py" --ablation_name static_only --branch_mode static_only

echo.
echo [4/8] Stress - Dynamic Branch Only
"%PYTHON_EXE%" "%~dp0train_stress_physics_mamba.py" --ablation_name dynamic_only --branch_mode dynamic_only

REM --- Subsidence Ablations ---

echo.
echo [5/8] Subsidence - Full Physics (Mamba Dual)
"%PYTHON_EXE%" "%~dp0train_subsidence_physics_mamba.py" --ablation_name full --branch_mode dual

echo.
echo [6/8] Subsidence - No Physics Loss (Mamba Dual)
"%PYTHON_EXE%" "%~dp0train_subsidence_physics_mamba.py" --ablation_name no_physics --branch_mode dual --no_physics

echo.
echo [7/8] Subsidence - Static Branch Only
"%PYTHON_EXE%" "%~dp0train_subsidence_physics_mamba.py" --ablation_name static_only --branch_mode static_only

echo.
echo [8/8] Subsidence - Dynamic Branch Only
"%PYTHON_EXE%" "%~dp0train_subsidence_physics_mamba.py" --ablation_name dynamic_only --branch_mode dynamic_only

echo.
echo =================================================================
echo  Ablation Studies Completed!
echo =================================================================
pause
