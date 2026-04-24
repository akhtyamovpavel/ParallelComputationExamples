#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --comment "CUDA cooperative groups: warp reduction via thread_block_tile<32>"

./main
