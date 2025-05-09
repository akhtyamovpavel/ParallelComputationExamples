cmake_minimum_required(VERSION 3.5)
project(OpenMPHelloWorld)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(hello_world ${SOURCE_FILES})
target_link_libraries(hello_world OpenMP::OpenMP_C)
