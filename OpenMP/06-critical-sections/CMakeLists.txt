cmake_minimum_required(VERSION 3.5)
project(OpenMPCriticalSections)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lrt")

add_executable(main_critical main_critical.cpp)
add_executable(main_atomic main_atomic.cpp)
add_executable(main_local_atomic main_local_atomic.cpp)
add_executable(main main.cpp)

target_link_libraries(main_critical OpenMP::OpenMP_C)
target_link_libraries(main_atomic OpenMP::OpenMP_C)
target_link_libraries(main_local_atomic OpenMP::OpenMP_C)
target_link_libraries(main OpenMP::OpenMP_C)
