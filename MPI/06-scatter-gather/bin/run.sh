#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

sbcast -f ./MpiScatterGather /tmp/$USER-MpiScatterGather
mpiexec /tmp/$USER-MpiScatterGather
