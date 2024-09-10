cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(MpiTestProbe)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiTestProbe_SOURCE_DIR}/bin)

add_executable(MpiWithoutProbe main.cpp)
add_executable(MpiProbe probe.cpp)
