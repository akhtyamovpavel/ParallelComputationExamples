#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --partition=RT
#SBATCH --job-name=example

mpiexec ./MpiHelloWorld
