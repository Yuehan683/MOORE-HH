@echo off
REM ================================
REM Windows version of run_minigrid_ppo_st.sh
REM Usage:
REM   run_minigrid_ppo_st.bat MiniGrid-DoorKey-6x6-v0
REM ================================

cd /d %~dp0\..\..\..
set ENV_NAME=%1

if "%ENV_NAME%"=="" (
    echo ERROR: ENV_NAME is not provided.
    echo Usage: run_minigrid_ppo_st.bat MiniGrid-DoorKey-6x6-v0
    exit /b 1
)

set RESULTS_DIR=F:\moore\MOORE\results
if not exist "%RESULTS_DIR%" (
    mkdir "%RESULTS_DIR%"
)

python run_minigrid_ppo_st_gpu.py ^
--n_exp 5 ^
--env_name %ENV_NAME% ^
--exp_name ppo_st_baseline ^
--results_dir "%RESULTS_DIR%" ^
--n_epochs 100 ^
--n_steps 2000 ^
--n_episodes_test 16 ^
--train_frequency 2000 ^
--lr_actor 1e-3 ^
--lr_critic 1e-3 ^
--critic_network MiniGridPPONetwork ^
--critic_n_features 128 ^
--actor_network MiniGridPPONetwork ^
--actor_n_features 128 ^
--batch_size 256 ^
--gamma 0.99 ^
--use_cuda
pause
