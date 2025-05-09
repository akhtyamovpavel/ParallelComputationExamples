cmake_minimum_required(VERSION 3.5)
project(OpenMPMaster)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(master ${SOURCE_FILES})
target_link_libraries(master OpenMP::OpenMP_C)
