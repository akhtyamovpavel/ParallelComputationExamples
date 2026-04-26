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

add_opencl_example(OpenCLPlatformInfo  00-platform-device-info           main.cpp)
add_opencl_example(OpenCLVectorAdd     01-intro/01-vector-add            main.cpp)
add_opencl_example(OpenCLTranspose     02-local-memory/00-transpose      main.cpp)
add_opencl_example(OpenCLReduction     03-reduction                      main.cpp)
add_opencl_example(OpenCLMatmulNaive   04-matrix-multiplication          main.cpp)
add_opencl_example(OpenCLProfiling     05-profiling-events               main.cpp)
add_opencl_example(OpenCLImageBlur     06-image-objects                  main.cpp)
add_opencl_example(OpenCLMultiQueue    07-multi-queue                    main.cpp)
add_opencl_example(OpenCLCoalesced     08-coalesced-access               main.cpp)
add_opencl_example(OpenCLConstant      09-constant-memory                main.cpp)
add_opencl_example(OpenCLPinned        10-pinned-memory                  main.cpp)
add_opencl_example(OpenCLStencil       11-stencil                        main.cpp)
add_opencl_example(OpenCLAtomics       12-atomics                        main.cpp)
add_opencl_example(OpenCLScan          13-scan                           main.cpp)
add_opencl_example(OpenCLKernelFile    14-kernel-from-file               main.cpp)
add_opencl_example(OpenCLBuildOptions  15-build-options                  main.cpp)
add_opencl_example(OpenCLOfflineComp   16-offline-compilation            main.cpp)
add_opencl_example(OpenCLSubGroups     17-sub-groups                     main.cpp)
add_opencl_example(OpenCLSvm          18-svm                             main.cpp)
add_opencl_example(OpenCLMultiDevice   19-multi-device                   main.cpp)
# 20-pyopencl — Python, не CMake target
