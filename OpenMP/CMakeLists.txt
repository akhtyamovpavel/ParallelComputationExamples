cmake_minimum_required(VERSION 3.10)
project(OpenMPExamples)

# Находим OpenMP
find_package(OpenMP REQUIRED)

# Добавляем все подпроекты
add_subdirectory(00-hello-world)
add_subdirectory(01-parallel-for)
add_subdirectory(02-sections)
add_subdirectory(03-master)
add_subdirectory(04-parralel-sum)
add_subdirectory(05-matrix-multiplication)
add_subdirectory(06-critical-sections) 