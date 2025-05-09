cmake_minimum_required(VERSION 3.10)
project(ParallelComputationExamples)

# Опции для включения/выключения разных частей проекта
option(BUILD_MPI_EXAMPLES "Build MPI examples" ON)
option(BUILD_OPENMP_EXAMPLES "Build OpenMP examples" ON)
option(BUILD_CUDA_EXAMPLES "Build CUDA examples" OFF)

# Добавляем подпроекты
if(BUILD_MPI_EXAMPLES)
    add_subdirectory(MPI)
endif()

if(BUILD_OPENMP_EXAMPLES)
    add_subdirectory(OpenMP)
endif()

if(BUILD_CUDA_EXAMPLES)
    add_subdirectory(CUDA)
endif()
