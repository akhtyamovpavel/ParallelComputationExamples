#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1

sbcast -f ./MpiCommGroup /tmp/$USER-MpiCommGroup
mpiexec /tmp/$USER-MpiCommGroup
