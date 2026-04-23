#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

sbcast -f ./MpiCommDup /tmp/$USER-MpiCommDup
mpiexec /tmp/$USER-MpiCommDup
