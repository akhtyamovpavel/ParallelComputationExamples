#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

executable=$1
sbcast -f $executable $PWD/$executable

mpiexec ./$executable
