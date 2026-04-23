#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

sbcast -f ./MpiAlltoall /tmp/$USER-MpiAlltoall
mpiexec /tmp/$USER-MpiAlltoall
