@echo off
echo ========================================================
echo   Starting Parallel Robustness Training (Mamba + Baselines)
echo   Seeds: 10, 15, 29, 53, 65, 77, 88, 92
echo ========================================================

:: Change to directory of this script
cd /d %~dp0

:: Create result directories if they don't exist (relative to project root via logical path handling in sub-script, 
:: but good to create here too relative to ..)
if not exist "robustness_results_stress" mkdir "robustness_results_stress"
if not exist "robustness_results_subsidence" mkdir "robustness_results_subsidence"
if not exist "robustness_results_stress_baselines" mkdir "robustness_results_stress_baselines"
if not exist "robustness_results_subsidence_baselines" mkdir "robustness_results_subsidence_baselines"

set SEEDS=10 15 29 53 65 77 88 92

:: Launch one window per seed
for %%s in (%SEEDS%) do (
    echo Launching training for SEED %%s...
    :: Start a new CMD window running the helper script
    start "Train_Seed_%%s" cmd /c "run_single_seed_train.bat %%s"
)

echo.
echo All training workers dispatched.
echo Check the individual windows for progress.
pause
