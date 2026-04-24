#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --comment "CUDA WMMA: minimal one-warp 16x16 matmul (needs sm_70+)"

./main
