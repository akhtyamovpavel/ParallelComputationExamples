#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1

sbcast -f ./MpiCommSplit /tmp/$USER-MpiCommSplit
mpiexec /tmp/$USER-MpiCommSplit
