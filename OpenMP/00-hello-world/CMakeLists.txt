cmake_minimum_required(VERSION 2.8)
project(HelloWorld)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

set(SOURCE_FILES main.cpp)
add_executable(HelloWorld ${SOURCE_FILES})
