cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(MpiCommDup)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiCommDup_SOURCE_DIR}/bin)

add_executable(MpiCommDup main.cpp)
