@echo off
REM ==========================================
REM MOORE-HH Debug Run (MiniGrid MT, SingleHead, GPU)
REM Usage:
REM   run_minigrid_ppo_mt_moore_hh_singlehead_gpu_debug.bat MT3 3
REM Example:
REM   run_minigrid_ppo_mt_moore_hh_singlehead_gpu_debug.bat MT3 3
REM ==========================================

cd /d %~dp0\..\..\..

set MT_NAME=%1
set N_EXPERTS=%2

if "%MT_NAME%"=="" (
    echo ERROR: MT_NAME is not provided.
    echo Usage: run_minigrid_ppo_mt_moore_hh_singlehead_gpu_debug.bat MT3 3
    exit /b 1
)

if "%N_EXPERTS%"=="" (
    echo ERROR: N_EXPERTS is not provided.
    echo Usage: run_minigrid_ppo_mt_moore_hh_singlehead_gpu_debug.bat MT3 3
    exit /b 1
)

python run_minigrid_ppo_mt.py ^
 --use_cuda ^
 --debug ^
 --n_exp 1 ^
 --seed 0 ^
 --env_name %MT_NAME% ^
 --exp_name debug_moore_hh_singlehead_%MT_NAME%_%N_EXPERTS%e ^
 --n_epochs 100 ^
 --n_steps 2000 ^
 --n_episodes_test 16 ^
 --train_frequency 2000 ^
 --lr_actor 1e-3 ^
 --lr_critic 1e-3 ^
 --critic_network MiniGridPPOMixtureSHNetwork ^
 --critic_n_features 128 ^
 --actor_network MiniGridPPOMixtureSHNetwork ^
 --actor_n_features 128 ^
 --orthogonal ^
 --hh_canon_sign ^
 --hh_rank_tol 1e-6 ^
 --n_experts %N_EXPERTS% ^
 --batch_size 256 ^
 --gamma 0.99

pause