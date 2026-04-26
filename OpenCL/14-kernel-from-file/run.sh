#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --comment "OpenCL kernel from .cl file"

./main
