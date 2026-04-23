#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1

sbcast -f ./MpiCommCart /tmp/$USER-MpiCommCart
mpiexec /tmp/$USER-MpiCommCart
