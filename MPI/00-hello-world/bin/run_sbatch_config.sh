#!/bin/bash
#
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=Sandbox
#SBATCH --job-name=example
#SBATCH --comment="Run mpi from config"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
sbcast -f MpiHelloWorld $(realpath MpiHelloWorld)
mpiexec $(realpath MpiHelloWorld)
