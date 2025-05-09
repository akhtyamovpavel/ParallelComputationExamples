cmake_minimum_required(VERSION 3.5)
project(OpenMPParallelFor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(parallel_for ${SOURCE_FILES})
target_link_libraries(parallel_for OpenMP::OpenMP_C)
