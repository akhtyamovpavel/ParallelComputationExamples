#!/bin/bash
sbcast -f ./MpiHelloWorld /home/$USER/MpiHelloWorld
mpiexec /home/$USER/MpiHelloWorld
