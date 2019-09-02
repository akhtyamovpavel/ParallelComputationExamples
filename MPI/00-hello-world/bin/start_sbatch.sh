#!/bin/bash
module add mpi/openmpi4-x86_64
sbatch -n 8 run.sh
sbatch -n 8 --ntasks-per-node 2 run.sh
sbatch run_sbatch_config.sh
