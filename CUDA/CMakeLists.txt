cmake_minimum_required(VERSION 3.18)
project(CudaExamples LANGUAGES CXX CUDA)

# Зонтичная сборка CUDA-примеров. Зеркалит подход MPI/CMakeLists.txt:
#  - каждый пример собирается как отдельный target;
#  - бинарник кладётся прямо в директорию примера под именем `main`,
#    чтобы соседний run.sh (и ручной `./main`) нашёл его там же,
#    где лежит исходник.
# Самостоятельная сборка (вручную `nvcc main.cu -o main`) продолжает работать.
#
# Архитектуры можно переопределить через -DCMAKE_CUDA_ARCHITECTURES=...
# По умолчанию берём основные поколения от Pascal до Ada Lovelace:
#   60 — Pascal P100
#   61 — Pascal consumer (GTX 10xx, Titan X Pascal — как на кластере)
#   70 — Volta V100
#   75 — Turing (T4, RTX 20xx)
#   80 — Ampere A100
#   86 — Ampere consumer (RTX 30xx)
#   89 — Ada Lovelace (RTX 40xx)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Добавляет пример, у которого внутри каталога ровно один файл main.cu.
# Имя target'а автоматически строится из пути (чтобы не конфликтовать внутри
# CMake), а OUTPUT_NAME всегда `main` — этого ждёт run.sh.
function(add_cuda_example subdir)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    string(REPLACE "/" "_" target "cuda_${subdir}")
    string(REPLACE "." "_" target "${target}")
    add_executable(${target} ${example_dir}/main.cu)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME main
        RUNTIME_OUTPUT_DIRECTORY ${example_dir})
endfunction()

# Для каталогов, в которых несколько .cu-файлов (02-shared, 00-stupid-sum,
# 05-scan и т.п.): даём каждому свой target и своё имя бинарника.
function(add_cuda_example_named target subdir source)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_executable(${target} ${example_dir}/${source})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${target}
        RUNTIME_OUTPUT_DIRECTORY ${example_dir})
endfunction()

# ----- 00 / 01 / 02 -----
add_cuda_example(00-how-to-run-task-on-cluster)

add_cuda_example(01-intro/01-hello-world-single-cuda-thread)
add_cuda_example(01-intro/02-add-threads)
add_cuda_example(01-intro/03-add-blocks)
add_cuda_example(01-intro/04-memcpy)
add_cuda_example(01-intro/05-matrix-multiplication-single)

add_cuda_example(02-device-specs-benchmarks/00-time-measurement)
add_cuda_example(02-device-specs-benchmarks/01-bandwidth-throughtput)
add_cuda_example(02-device-specs-benchmarks/02-ilp)
add_cuda_example(02-device-specs-benchmarks/03-occupancy)

# ----- 03-memory-model -----
add_cuda_example_named(Uncoalesced    03-memory-model/01-uncoalesced-access  full_uncoalesce.cu)
add_cuda_example_named(SharedGlobal   03-memory-model/02-shared              global_example.cu)
add_cuda_example_named(SharedLocal    03-memory-model/02-shared              shared_example.cu)
add_cuda_example(03-memory-model/03-host-alloc-benchmarks)
add_cuda_example(03-memory-model/04-caching)
add_cuda_example(03-memory-model/05-pitching)

add_cuda_example(03-memory-model/06-constant/00-global)
add_cuda_example(03-memory-model/06-constant/01-constant)

add_cuda_example(03-memory-model/07-transpose/00-naive)
add_cuda_example(03-memory-model/07-transpose/01-tiled)
add_cuda_example(03-memory-model/07-transpose/02-tiled-padded)

add_cuda_example(03-memory-model/08-stencil/00-global)
add_cuda_example(03-memory-model/08-stencil/01-shared-halo)

add_cuda_example(03.5-matrix-multiplication-example)
# Tiled matmul через shared memory — это ДЗ студентов (tasks/02-matrix-multiplication.md),
# пример намеренно НЕ поставляется.

# ----- 04-reduction (несколько директорий с множеством .cu) -----
add_cuda_example_named(ReduceStupid     04-reduction/00-stupid-sum    main.cu)
add_cuda_example_named(ReduceStupidNorm 04-reduction/00-stupid-sum    normal_sum.cu)
add_cuda_example_named(ReduceDefault    04-reduction/01-default-sum   main.cu)
add_cuda_example_named(ReduceNoShared   04-reduction/01-default-sum   main_without_shared_memory.cu)
add_cuda_example(04-reduction/02-another-math)
add_cuda_example(04-reduction/03-improving-bank-conflicts)
add_cuda_example(04-reduction/04-add-on-load)
add_cuda_example(04-reduction/05-warp-reduce)
add_cuda_example(04-reduction/06-warp-design-specific)

# ----- 04.5 / 04.7 (добавлены в рамках ROADMAP) -----
add_cuda_example(04.5-atomics/00-atomic-add)
add_cuda_example(04.5-atomics/01-histogram-naive)
add_cuda_example(04.5-atomics/02-histogram-privatized)

add_cuda_example(04.7-warp-primitives/00-warp-shuffle)
add_cuda_example(04.7-warp-primitives/01-warp-reduce)
add_cuda_example(04.7-warp-primitives/02-warp-vote)

# ----- 05-scan (плоская директория с несколькими .cu) -----
add_cuda_example_named(ScanNaiveCpu       05-scan  00-naive_scan.cu)
add_cuda_example_named(ScanNaiveCuda      05-scan  01-naive_cuda_scan.cu)
add_cuda_example_named(ScanCorrect        05-scan  01.5-correct_cuda_scan.cu)
add_cuda_example_named(ScanRecursive      05-scan  01.75-recursive_cuda_scan.cu)
add_cuda_example_named(ScanTwoLayer       05-scan  02-two-layer-parallelism.cu)
add_cuda_example_named(ScanBankConflicts  05-scan  03-scan-bank-conflicts.cu)

# ----- 06-cublas (нужна линковка с cuBLAS) -----
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    function(add_cublas_example target subdir source)
        set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        add_executable(${target} ${example_dir}/${source})
        target_link_libraries(${target} PRIVATE CUDA::cublas)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME ${target}
            RUNTIME_OUTPUT_DIRECTORY ${example_dir})
    endfunction()
    add_cublas_example(CublasVectorAdd    06-cublas  01-vector-add.cu)
    add_cublas_example(CublasVectorAdding 06-cublas  02-vector-adding.cu)
    add_cublas_example(CublasCosine       06-cublas  03-cosine-distance.cu)
    add_cublas_example(CublasMatrixSums   06-cublas  04-matrix-sums.cu)
else()
    message(WARNING "CUDAToolkit не найден: примеры 06-cublas пропущены")
endif()

# 07-pycuda — это Python-пример, не target для CMake.

# ----- 08 / 09 (добавлены в рамках ROADMAP) -----
add_cuda_example(08-streams-events/00-events-timing)
add_cuda_example(08-streams-events/01-async-memcpy)
add_cuda_example(08-streams-events/02-overlap-compute-copy)

add_cuda_example(09-unified-memory/00-managed-basic)
add_cuda_example(09-unified-memory/01-prefetch)

# ----- 10-cooperative-groups (добавлены в рамках ROADMAP) -----
add_cuda_example(10-cooperative-groups/00-warp-reduce-cg)

# ----- 11-cuda-graphs (добавлены в рамках ROADMAP) -----
add_cuda_example(11-cuda-graphs/00-stream-capture)
add_cuda_example(11-cuda-graphs/01-explicit-graph)

# ----- 12-thrust (добавлены в рамках ROADMAP, header-only) -----
add_cuda_example(12-thrust/00-device-vector)
add_cuda_example(12-thrust/01-transform-reduce)
add_cuda_example(12-thrust/02-sort)
# ----- 14-multi-gpu (нужны минимум 2 GPU для запуска; сборка идёт везде) -----
add_cuda_example(14-multi-gpu/00-independent-devices)
add_cuda_example(14-multi-gpu/01-peer-access)

# ----- 16-dynamic-parallelism (требует -rdc=true + cudadevrt) -----
function(add_cdp_example subdir)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    string(REPLACE "/" "_" target "cuda_${subdir}")
    string(REPLACE "." "_" target "${target}")
    add_executable(${target} ${example_dir}/main.cu)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME main
        RUNTIME_OUTPUT_DIRECTORY ${example_dir}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()
add_cdp_example(16-dynamic-parallelism/00-child-launch)

# ----- 15-tensor-cores (требуют sm_70+, переопределяем архитектуры target'а) -----
function(add_wmma_example subdir)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    string(REPLACE "/" "_" target "cuda_${subdir}")
    string(REPLACE "." "_" target "${target}")
    add_executable(${target} ${example_dir}/main.cu)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME main
        RUNTIME_OUTPUT_DIRECTORY ${example_dir}
        CUDA_ARCHITECTURES "70;75;80;86;89")
endfunction()
add_wmma_example(15-tensor-cores/00-wmma-16x16)
add_wmma_example(15-tensor-cores/01-wmma-tiled)

# ----- 13-curand / 17-cufft / 18-cusparse (все линкуются с тулкитовыми либами) -----
# find_package(CUDAToolkit) уже вызывался выше для cuBLAS; повторный вызов
# дешёвый — результат кешируется.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    function(add_curand_example subdir)
        set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        string(REPLACE "/" "_" target "cuda_${subdir}")
        string(REPLACE "." "_" target "${target}")
        add_executable(${target} ${example_dir}/main.cu)
        target_link_libraries(${target} PRIVATE CUDA::curand)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME main
            RUNTIME_OUTPUT_DIRECTORY ${example_dir})
    endfunction()
    add_curand_example(13-curand/00-monte-carlo-host-api)
    add_curand_example(13-curand/01-monte-carlo-device-api)

    function(add_cufft_example subdir)
        set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        string(REPLACE "/" "_" target "cuda_${subdir}")
        string(REPLACE "." "_" target "${target}")
        add_executable(${target} ${example_dir}/main.cu)
        target_link_libraries(${target} PRIVATE CUDA::cufft)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME main
            RUNTIME_OUTPUT_DIRECTORY ${example_dir})
    endfunction()
    add_cufft_example(17-cufft/00-roundtrip)

    function(add_cusparse_example subdir)
        set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
        string(REPLACE "/" "_" target "cuda_${subdir}")
        string(REPLACE "." "_" target "${target}")
        add_executable(${target} ${example_dir}/main.cu)
        target_link_libraries(${target} PRIVATE CUDA::cusparse)
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME main
            RUNTIME_OUTPUT_DIRECTORY ${example_dir})
    endfunction()
    add_cusparse_example(18-cusparse/00-spmv)
else()
    message(WARNING "CUDAToolkit не найден: 13-curand, 17-cufft, 18-cusparse пропущены")
endif()
