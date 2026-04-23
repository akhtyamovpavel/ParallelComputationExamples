#!/bin/bash
mpiexec -np 4 ./MpiCollectiveBenchmark "$@" | tee results.csv
