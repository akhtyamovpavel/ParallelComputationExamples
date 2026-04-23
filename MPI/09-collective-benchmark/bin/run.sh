#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --output=bench-%j.csv

sbcast -f ./MpiCollectiveBenchmark /tmp/$USER-MpiCollectiveBenchmark
mpiexec /tmp/$USER-MpiCollectiveBenchmark
