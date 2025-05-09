cmake_minimum_required(VERSION 3.5)
project(OpenMPParallelSum)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(parallel_sum ${SOURCE_FILES})
target_link_libraries(parallel_sum OpenMP::OpenMP_C)
