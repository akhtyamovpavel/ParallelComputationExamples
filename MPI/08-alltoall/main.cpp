#include <mpi.h>
#include <cstdio>
#include <cstdlib>
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
            for (int i = 0; i < chunk; ++i)
                recvbuf[rank * chunk + i] = sendbuf[rank * chunk + i];
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
    for (int i = 0; i < 3; ++i) fn();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) fn();
    double t1 = MPI_Wtime();
    double local = (t1 - t0) / iters, worst;
    MPI_Reduce(&local, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%-20s time=%8.3f ms  bw=%7.2f MB/s (на один ранг)\n",
               name, worst * 1e3, bytes / worst / (1024.0 * 1024.0));
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
    std::vector<int> send(size * chunk), recv(size * chunk);
    for (int r = 0; r < size; ++r)
        for (int i = 0; i < chunk; ++i)
            send[r * chunk + i] = rank * 1000 + r;

    if (rank == 0) {
        printf("=== Alltoall, chunk=%d int на пира, ranks=%d ===\n",
               chunk, size);
    }

    double bytes = (double)size * chunk * sizeof(int);

    bench("MPI_Alltoall", rank, iters, bytes, [&] {
        MPI_Alltoall(send.data(), chunk, MPI_INT,
                     recv.data(), chunk, MPI_INT, MPI_COMM_WORLD);
    });
    bench("manual_alltoall", rank, iters, bytes, [&] {
        manual_alltoall(send.data(), recv.data(), chunk, MPI_COMM_WORLD);
    });

    // Проверка: в recv[r*chunk] должно быть (r*1000 + rank).
    int ok = 1;
    for (int r = 0; r < size; ++r)
        if (recv[r * chunk] != r * 1000 + rank) { ok = 0; break; }
    int all_ok;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("проверка: %s\n", all_ok ? "OK" : "FAIL");

    MPI_Finalize();
    return 0;
}
