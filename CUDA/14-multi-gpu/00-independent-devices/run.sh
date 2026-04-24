#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:2
#SBATCH --comment "CUDA multi-GPU: independent SAXPY on 2 devices (needs >=2 GPUs)"

./main
