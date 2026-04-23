cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(MpiReduceAllreduce)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiReduceAllreduce_SOURCE_DIR}/bin)

add_executable(MpiReduceAllreduce main.cpp)
