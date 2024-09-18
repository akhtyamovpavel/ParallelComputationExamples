#!/bin/bash
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1

sbcast -f MpiIsendRecv /tmp/$USER-MpiIsendRecv
mpiexec /tmp/$USER-MpiIsendRecv
