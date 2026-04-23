#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

sbcast -f ./MpiBcast /tmp/$USER-MpiBcast
mpiexec /tmp/$USER-MpiBcast
