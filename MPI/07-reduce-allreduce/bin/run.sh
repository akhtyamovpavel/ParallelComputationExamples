#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

sbcast -f ./MpiReduceAllreduce /tmp/$USER-MpiReduceAllreduce
mpiexec /tmp/$USER-MpiReduceAllreduce
