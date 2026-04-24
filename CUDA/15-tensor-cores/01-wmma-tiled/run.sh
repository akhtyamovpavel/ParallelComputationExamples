#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --comment "CUDA WMMA: tiled 1024x1024 matmul FP16 x FP16 -> FP32 (needs sm_70+)"

./main
