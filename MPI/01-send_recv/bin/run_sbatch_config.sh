#!/bin/bash
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=RT
#SBATCH --job-name=sendrecvexample
#SBATCH --comment="Run MPI SendRecv"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
mpiexec ./MpiSendRecv
