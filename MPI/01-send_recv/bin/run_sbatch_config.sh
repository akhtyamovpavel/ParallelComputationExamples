#!/bin/bash
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sendrecvexample
#SBATCH --comment="Run MPI SendRecv"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
sbcast -f MpiSendRecv $PWD/MpiSendRecv
ls $PWD
mpiexec ./MpiSendRecv
