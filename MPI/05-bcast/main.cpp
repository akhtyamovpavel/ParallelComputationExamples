#include <mpi.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Линейный broadcast: root последовательно отправляет сообщение каждому
// процессу. Время передачи ~ (P-1) * latency. Полезно как baseline.
static void linear_bcast(void* buf, int count, MPI_Datatype dt,
                         int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        for (int r = 0; r < size; ++r) {
            if (r == root) {
                continue;
            }
            MPI_Send(buf, count, dt, r, 0, comm);
        }
    } else {
        MPI_Recv(buf, count, dt, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

// Биномиальное дерево, сложность ~ log2(P) * latency.
// Используем виртуальные ранги: vrank = (rank - root + P) mod P.
// На шаге mask = 1, 2, 4, ...: процесс с vrank < mask шлёт соседу vrank + mask;
// процесс mask <= vrank < 2*mask принимает от vrank - mask.
static void tree_bcast(void* buf, int count, MPI_Datatype dt,
                       int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    int vrank = (rank - root + size) % size;
    int mask = 1;
    while (mask < size) {
        if (vrank < mask) {
            int peer = vrank + mask;
            if (peer < size) {
                MPI_Send(buf, count, dt, (peer + root) % size, 0, comm);
            }
        } else if (vrank < 2 * mask) {
            MPI_Recv(buf, count, dt, (vrank - mask + root) % size, 0,
                     comm, MPI_STATUS_IGNORE);
        }
        mask <<= 1;
    }
}

static void wrap_mpi_bcast(void* buf, int count, MPI_Datatype dt,
                           int root, MPI_Comm comm) {
    MPI_Bcast(buf, count, dt, root, comm);
}

typedef void (*bcast_fn)(void*, int, MPI_Datatype, int, MPI_Comm);

static void bench(const char* name, bcast_fn fn,
                  std::vector<int>& buf, int iters) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < 5; ++i) {
        fn(buf.data(), buf.size(), MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        fn(buf.data(), buf.size(), MPI_INT, 0, MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();
    double local = (t1 - t0) / iters;
    double worst = 0.0;
    MPI_Reduce(&local, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double bytes = buf.size() * static_cast<double>(sizeof(int));
        std::cout << std::left << std::setw(12) << name
                  << " bytes=" << std::left << std::setw(10)
                  << std::fixed << std::setprecision(0) << bytes
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
    int count = (argc > 1) ? atoi(argv[1]) : (1 << 18);  // ~1 MiB по умолчанию
    int iters = (argc > 2) ? atoi(argv[2]) : 50;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> buf(count);
    if (rank == 0) {
        for (int i = 0; i < count; ++i) {
            buf[i] = i;
        }
        std::cout << "=== Bcast: MPI_Bcast vs linear vs tree ===" << std::endl;
        std::cout << "ranks=" << size << " count=" << count
                  << " iters=" << iters << std::endl;
    }

    bench("MPI_Bcast", wrap_mpi_bcast, buf, iters);
    bench("linear",    linear_bcast,   buf, iters);
    bench("tree",      tree_bcast,     buf, iters);

    MPI_Finalize();
    return 0;
}
