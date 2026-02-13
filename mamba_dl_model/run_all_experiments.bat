@echo off
echo =================================================================
echo  STARTING FULL EXPERIMENTAL SUITE - Baselines and Ablations
echo =================================================================

echo Calling Baseline Script...
call "%~dp0run_all_mamba_baselines.bat"

echo Calling Ablation Script...
call "%~dp0run_all_mamba_ablations.bat"

echo.
echo =================================================================
echo  ALL EXPERIMENTS COMPLETED SUCCESSFULLY.
echo =================================================================
pause
