#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

sbcast -f MpiPingPong $PWD/MpiPingPong

mpiexec ./MpiPingPong
