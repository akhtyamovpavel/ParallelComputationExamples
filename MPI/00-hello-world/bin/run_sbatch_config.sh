#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=RT
#SBATCH --job-name=example
#SBATCH --comment="Run mpi from config"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
mpiexec ./MpiHelloWorld
