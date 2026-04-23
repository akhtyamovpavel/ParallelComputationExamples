cmake_minimum_required(VERSION 3.10)
project(MpiExamples CXX)

# В агрегированной сборке нельзя делать add_subdirectory в каждый пример:
# их CMakeLists.txt переопределяют CMAKE_CXX_COMPILER через set(), что
# конфликтует с уже настроенным тулчейном верхнего уровня. Поэтому здесь
# находим MPI через find_package и строим все примеры напрямую, линкуя
# их с MPI::MPI_CXX. Самостоятельные сборки каждого примера
# (cd MPI/XX && cmake . && make) продолжают работать.

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 14)

function(add_mpi_example target subdir source)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_executable(${target} ${example_dir}/${source})
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${example_dir}/bin)
endfunction()

add_mpi_example(MpiHelloWorld          00-hello-world           main.cpp)
add_mpi_example(MpiSendRecv            01-send_recv             main.cpp)
add_mpi_example(MpiPingPong            02-ping-pong             main.cpp)
add_mpi_example(MpiWithoutProbe        03-probe-message-status  main.cpp)
add_mpi_example(MpiProbe               03-probe-message-status  probe.cpp)
add_mpi_example(MpiIsendRecv           04-isend-irecv           main.cpp)
add_mpi_example(MpiBcast               05-bcast                 main.cpp)
add_mpi_example(MpiScatterGather       06-scatter-gather        main.cpp)
add_mpi_example(MpiReduceAllreduce     07-reduce-allreduce      main.cpp)
add_mpi_example(MpiAlltoall            08-alltoall              main.cpp)
add_mpi_example(MpiCollectiveBenchmark 09-collective-benchmark  main.cpp)
add_mpi_example(MpiCommSplit           10-comm-split            main.cpp)
add_mpi_example(MpiCommDup             11-comm-dup              main.cpp)
add_mpi_example(MpiCommGroup           12-comm-group            main.cpp)
add_mpi_example(MpiCommCart            13-comm-cart             main.cpp)
