#include <mpi.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Ручная редукция по биномиальному дереву к root (суммирование).
// На шаге mask = 1, 2, 4, ...:
//   если (vrank & mask) == 0 — принимаем от vrank|mask и добавляем;
//   иначе шлём накопленный вектор к vrank & ~mask и выходим.
static void tree_reduce_sum(const double* sendbuf, double* recvbuf,
                            int count, int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    std::vector<double> work(sendbuf, sendbuf + count);
    std::vector<double> tmp(count);
    int vrank = (rank - root + size) % size;
    int mask = 1;
    while (mask < size) {
        if ((vrank & mask) == 0) {
            int peer = vrank | mask;
            if (peer < size) {
                MPI_Recv(tmp.data(), count, MPI_DOUBLE,
                         (peer + root) % size, 0, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; ++i) {
                    work[i] += tmp[i];
                }
            }
        } else {
            int dst_v = vrank & ~mask;
            MPI_Send(work.data(), count, MPI_DOUBLE,
                     (dst_v + root) % size, 0, comm);
            return;
        }
        mask <<= 1;
    }
    if (rank == root) {
        for (int i = 0; i < count; ++i) {
            recvbuf[i] = work[i];
        }
    }
}

template <typename F>
static void bench(const char* name, int rank, int iters, double bytes, F fn) {
    for (int i = 0; i < 3; ++i) {
        fn();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    double t1 = MPI_Wtime();
    double local = (t1 - t0) / iters;
    double worst = 0.0;
    MPI_Reduce(&local, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << std::left << std::setw(20) << name
                  << " time=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << worst * 1e3 << " ms"
                  << "  bw=" << std::right << std::setw(7)
                  << std::fixed << std::setprecision(2)
                  << bytes / worst / (1024.0 * 1024.0) << " MB/s"
                  << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = (argc > 1) ? atoi(argv[1]) : (1 << 16);
    int iters = (argc > 2) ? atoi(argv[2]) : 50;

    // У каждого ранга свой вектор [1+rank, 1+rank, ...] — легко проверить сумму.
    std::vector<double> local(count, 1.0 + rank);
    std::vector<double> result(count, 0.0);

    if (rank == 0) {
        std::cout << "=== Reduce / Allreduce, count=" << count
                  << ", ranks=" << size << " ===" << std::endl;
    }

    double bytes = count * static_cast<double>(sizeof(double));

    bench("MPI_Reduce", rank, iters, bytes, [&] {
        MPI_Reduce(local.data(), result.data(), count, MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);
    });
    bench("tree_reduce_sum", rank, iters, bytes, [&] {
        tree_reduce_sum(local.data(), result.data(), count, 0, MPI_COMM_WORLD);
    });
    bench("MPI_Allreduce", rank, iters, bytes, [&] {
        MPI_Allreduce(local.data(), result.data(), count, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    });

    // Проверка корректности после Allreduce (у всех тот же результат)
    double expected = 0.0;
    for (int r = 0; r < size; ++r) {
        expected += 1.0 + r;
    }
    if (rank == 0) {
        std::cout << "result[0] = "
                  << std::fixed << std::setprecision(1) << result[0]
                  << " (ожидается " << expected << ")" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
