cmake_minimum_required(VERSION 3.10)
project(OpenCLExamples CXX)

# Зонтичная сборка OpenCL-примеров, зеркалит MPI/CMakeLists.txt и
# CUDA/CMakeLists.txt. Каждый пример строится как отдельный target,
# бинарник кладётся в директорию примера под именем `main` — это то,
# чего ожидает соседний run.sh.

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 14)

function(add_opencl_example target subdir source)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_executable(${target} ${example_dir}/${source})
    target_link_libraries(${target} PRIVATE OpenCL::OpenCL)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME main
        RUNTIME_OUTPUT_DIRECTORY ${example_dir})
endfunction()

add_opencl_example(OpenCLPlatformInfo  00-platform-device-info  main.cpp)
add_opencl_example(OpenCLVectorAdd     01-intro/01-vector-add   main.cpp)
