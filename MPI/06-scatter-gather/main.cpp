#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
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
                  << total_bytes / worst / (1024.0 * 1024.0) << " MB/s"
                  << std::endl;
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
        std::cout << "=== Scatter / Gather, chunk=" << chunk
                  << " int, ranks=" << size << " ===" << std::endl;
    }
    std::vector<int> local(chunk);
    std::vector<int> gathered(rank == 0 ? total : 0);
    double total_bytes = static_cast<double>(total) * sizeof(int);

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
        std::cout << "\n--- MPI_Scatterv: процесс r получает r+1 элементов ---"
                  << std::endl;
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
    std::ostringstream oss;
    oss << "[rank " << rank << "] получил " << counts[rank] << " элементов:";
    for (int i = 0; i < counts[rank]; ++i) {
        oss << " " << dst[i];
    }
    std::string line = oss.str();

    if (rank == 0) {
        std::cout << line << std::endl;
        for (int r = 1; r < size; ++r) {
            int incoming = 0;
            MPI_Status st;
            MPI_Probe(r, 42, MPI_COMM_WORLD, &st);
            MPI_Get_count(&st, MPI_CHAR, &incoming);
            std::vector<char> buf(incoming);
            MPI_Recv(buf.data(), incoming, MPI_CHAR, r, 42,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << std::string(buf.begin(), buf.end()) << std::endl;
        }
    } else {
        MPI_Send(line.data(), static_cast<int>(line.size()), MPI_CHAR,
                 0, 42, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
