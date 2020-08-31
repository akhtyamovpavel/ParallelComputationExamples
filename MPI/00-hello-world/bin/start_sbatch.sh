#!/bin/bash
module add mpi/openmpi4-x86_64
sbatch -n 8 --comment="Hello world on MPI" run.sh
sbatch -n 8 --comment="Hello world on MPI per two nodes" run.sh
sbatch run_sbatch_config.sh
