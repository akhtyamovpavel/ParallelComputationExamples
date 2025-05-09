cmake_minimum_required(VERSION 3.5)
project(OpenMPSections)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(sections ${SOURCE_FILES})
target_link_libraries(sections OpenMP::OpenMP_C)
