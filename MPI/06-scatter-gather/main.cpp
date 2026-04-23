#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>

// Ручной Scatter: root копирует себе свой кусок и раздаёт остальным Send-ами.
static void manual_scatter(const int* sendbuf, int* recvbuf, int chunk,
                           int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        for (int r = 0; r < size; ++r) {
            if (r == root) {
                std::copy(sendbuf + r * chunk, sendbuf + (r + 1) * chunk, recvbuf);
            } else {
                MPI_Send(sendbuf + r * chunk, chunk, MPI_INT, r, 0, comm);
            }
        }
    } else {
        MPI_Recv(recvbuf, chunk, MPI_INT, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

// Ручной Gather: каждый шлёт root-у свой кусок.
static void manual_gather(const int* sendbuf, int* recvbuf, int chunk,
                          int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        for (int r = 0; r < size; ++r) {
            if (r == root) {
                std::copy(sendbuf, sendbuf + chunk, recvbuf + r * chunk);
            } else {
                MPI_Recv(recvbuf + r * chunk, chunk, MPI_INT, r, 0,
                         comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Send(sendbuf, chunk, MPI_INT, root, 0, comm);
    }
}

template <typename F>
static void bench(const char* name, int rank, int iters, double total_bytes, F fn) {
    for (int i = 0; i < 3; ++i) fn();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) fn();
    double t1 = MPI_Wtime();
    double local = (t1 - t0) / iters, worst;
    MPI_Reduce(&local, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%-20s time=%8.3f ms  bw=%7.2f MB/s\n", name,
               worst * 1e3, total_bytes / worst / (1024.0 * 1024.0));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunk = (argc > 1) ? atoi(argv[1]) : (1 << 16);  // 64K int на процесс
    int iters = (argc > 2) ? atoi(argv[2]) : 30;
    int total = chunk * size;

    std::vector<int> full;
    if (rank == 0) {
        full.resize(total);
        std::iota(full.begin(), full.end(), 0);
        printf("=== Scatter / Gather, chunk=%d int, ranks=%d ===\n", chunk, size);
    }
    std::vector<int> local(chunk), gathered(rank == 0 ? total : 0);
    double total_bytes = (double)total * sizeof(int);

    bench("MPI_Scatter", rank, iters, total_bytes, [&] {
        MPI_Scatter(full.data(), chunk, MPI_INT,
                    local.data(), chunk, MPI_INT, 0, MPI_COMM_WORLD);
    });
    bench("manual_scatter", rank, iters, total_bytes, [&] {
        manual_scatter(full.data(), local.data(), chunk, 0, MPI_COMM_WORLD);
    });
    bench("MPI_Gather", rank, iters, total_bytes, [&] {
        MPI_Gather(local.data(), chunk, MPI_INT,
                   gathered.data(), chunk, MPI_INT, 0, MPI_COMM_WORLD);
    });
    bench("manual_gather", rank, iters, total_bytes, [&] {
        manual_gather(local.data(), gathered.data(), chunk, 0, MPI_COMM_WORLD);
    });

    // Пример MPI_Scatterv: неровная раздача. Ранг r получает (r+1) элементов.
    if (rank == 0) {
        printf("\n--- MPI_Scatterv: процесс r получает r+1 элементов ---\n");
    }
    std::vector<int> counts(size), displs(size);
    int running = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = r + 1;
        displs[r] = running;
        running += counts[r];
    }
    std::vector<int> src;
    if (rank == 0) {
        src.resize(running);
        std::iota(src.begin(), src.end(), 100);
    }
    std::vector<int> dst(counts[rank]);
    MPI_Scatterv(src.data(), counts.data(), displs.data(), MPI_INT,
                 dst.data(), counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Собираем строки обратно к root, чтобы не перепутать вывод нескольких процессов.
    char line[256];
    int n = snprintf(line, sizeof(line), "[rank %d] получил %d элементов:", rank, counts[rank]);
    for (int i = 0; i < counts[rank] && n < (int)sizeof(line) - 8; ++i) {
        n += snprintf(line + n, sizeof(line) - n, " %d", dst[i]);
    }
    if (rank == 0) {
        printf("%s\n", line);
        for (int r = 1; r < size; ++r) {
            char buf[256];
            MPI_Recv(buf, sizeof(buf), MPI_CHAR, r, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", buf);
        }
    } else {
        MPI_Send(line, n + 1, MPI_CHAR, 0, 42, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
