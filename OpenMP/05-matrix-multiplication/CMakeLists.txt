cmake_minimum_required(VERSION 3.5)
project(OpenMPMatrixMult)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(matrix_multiplication ${SOURCE_FILES})
target_link_libraries(matrix_multiplication OpenMP::OpenMP_C)
