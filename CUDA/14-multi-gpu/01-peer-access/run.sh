#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:2
#SBATCH --comment "CUDA multi-GPU peer access: GPU 1 kernel reads GPU 0 buffer"

./main
