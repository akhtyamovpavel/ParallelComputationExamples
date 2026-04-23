#include <mpi.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Ручной Alltoall через Sendrecv по всем парам.
// Простая "прямая" схема — каждый процесс для каждого пира делает обмен.
// Работает, но проигрывает оптимизированным схемам MPI (pairwise, Bruck).
static void manual_alltoall(const int* sendbuf, int* recvbuf, int chunk,
                            MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            for (int i = 0; i < chunk; ++i) {
                recvbuf[rank * chunk + i] = sendbuf[rank * chunk + i];
            }
        } else {
            MPI_Sendrecv(
                sendbuf + r * chunk, chunk, MPI_INT, r, 0,
                recvbuf + r * chunk, chunk, MPI_INT, r, 0,
                comm, MPI_STATUS_IGNORE);
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
                  << bytes / worst / (1024.0 * 1024.0)
                  << " MB/s (на один ранг)" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = (argc > 1) ? atoi(argv[1]) : 1024;  // int на каждого пира
    int iters = (argc > 2) ? atoi(argv[2]) : 100;

    // send[r*chunk..] — данные, которые rank отправит процессу r.
    // Помечаем (src=rank, dst=r), чтобы после обмена проверить корректность.
    std::vector<int> send(size * chunk);
    std::vector<int> recv(size * chunk);
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < chunk; ++i) {
            send[r * chunk + i] = rank * 1000 + r;
        }
    }

    if (rank == 0) {
        std::cout << "=== Alltoall, chunk=" << chunk
                  << " int на пира, ranks=" << size << " ===" << std::endl;
    }

    double bytes = static_cast<double>(size) * chunk * sizeof(int);

    bench("MPI_Alltoall", rank, iters, bytes, [&] {
        MPI_Alltoall(send.data(), chunk, MPI_INT,
                     recv.data(), chunk, MPI_INT, MPI_COMM_WORLD);
    });
    bench("manual_alltoall", rank, iters, bytes, [&] {
        manual_alltoall(send.data(), recv.data(), chunk, MPI_COMM_WORLD);
    });

    // Проверка: в recv[r*chunk] должно быть (r*1000 + rank).
    int ok = 1;
    for (int r = 0; r < size; ++r) {
        if (recv[r * chunk] != r * 1000 + rank) {
            ok = 0;
            break;
        }
    }
    int all_ok = 0;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "проверка: " << (all_ok ? "OK" : "FAIL") << std::endl;
    }

    MPI_Finalize();
    return 0;
}
