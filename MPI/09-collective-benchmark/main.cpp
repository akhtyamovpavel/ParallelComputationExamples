#include <mpi.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Сводный бенчмарк коллективов: проходит по размерам сообщения и печатает
// CSV-строки (collective,bytes,ranks,time_ms,bandwidth_MB_s).
// Для каждой точки — WARMUP прогонов, потом ITERS замеров.
// Время — max по всем рангам (худший процесс определяет реальное время операции).

static const int WARMUP = 5;
static const int ITERS  = 30;

template <typename F>
static double time_op(F fn) {
    for (int i = 0; i < WARMUP; ++i) {
        fn();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < ITERS; ++i) {
        fn();
    }
    double t1 = MPI_Wtime();
    double local = (t1 - t0) / ITERS;
    double worst = 0.0;
    MPI_Reduce(&local, &worst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return worst;  // достоверно только у rank 0
}

static void print_row(const char* name, double bytes, int ranks,
                      double time_s) {
    double ms = time_s * 1e3;
    double bw = bytes / time_s / (1024.0 * 1024.0);
    std::cout << name << ','
              << std::fixed << std::setprecision(0) << bytes << ','
              << ranks << ','
              << std::fixed << std::setprecision(4) << ms << ','
              << std::fixed << std::setprecision(2) << bw
              << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размеры сообщения в байтах. Можно переопределить: argv[1] — макс. байт.
    long max_bytes = (argc > 1) ? atol(argv[1]) : (8L << 20);
    std::vector<long> sizes;
    for (long s = 8; s <= max_bytes; s *= 8) {
        sizes.push_back(s);
    }

    if (rank == 0) {
        std::cout << "collective,bytes,ranks,time_ms,bandwidth_MB_s" << std::endl;
    }

    for (long nbytes : sizes) {
        // Bcast: сообщение размера nbytes.
        {
            std::vector<char> buf(nbytes, 'a');
            double t = time_op([&] {
                MPI_Bcast(buf.data(), nbytes, MPI_BYTE, 0, MPI_COMM_WORLD);
            });
            if (rank == 0) {
                print_row("Bcast", static_cast<double>(nbytes), size, t);
            }
        }

        // Reduce / Allreduce: эквивалентный объём в int-ах для MPI_SUM.
        int nint = static_cast<int>(std::max(1L, nbytes / static_cast<long>(sizeof(int))));
        double rb = static_cast<double>(nint) * sizeof(int);
        std::vector<int> isbuf(nint, 1);
        std::vector<int> irbuf(nint, 0);
        {
            double t = time_op([&] {
                MPI_Reduce(isbuf.data(), irbuf.data(), nint, MPI_INT,
                           MPI_SUM, 0, MPI_COMM_WORLD);
            });
            if (rank == 0) {
                print_row("Reduce", rb, size, t);
            }
        }
        {
            double t = time_op([&] {
                MPI_Allreduce(isbuf.data(), irbuf.data(), nint, MPI_INT,
                              MPI_SUM, MPI_COMM_WORLD);
            });
            if (rank == 0) {
                print_row("Allreduce", rb, size, t);
            }
        }

        // Alltoall: каждый ранг шлёт chunk байт каждому другому.
        long chunk = std::max(1L, nbytes / size);
        std::vector<char> as(chunk * size, 'x');
        std::vector<char> ar(chunk * size, 0);
        double ab = static_cast<double>(chunk) * size;
        {
            double t = time_op([&] {
                MPI_Alltoall(as.data(), chunk, MPI_BYTE,
                             ar.data(), chunk, MPI_BYTE, MPI_COMM_WORLD);
            });
            if (rank == 0) {
                print_row("Alltoall", ab, size, t);
            }
        }

        // Allgather: каждый отдаёт nbytes байт, все получают size*nbytes.
        std::vector<char> gs(nbytes, 'y');
        std::vector<char> gr(static_cast<size_t>(nbytes) * size, 0);
        {
            double t = time_op([&] {
                MPI_Allgather(gs.data(), nbytes, MPI_BYTE,
                              gr.data(), nbytes, MPI_BYTE, MPI_COMM_WORLD);
            });
            double total = static_cast<double>(nbytes) * size;
            if (rank == 0) {
                print_row("Allgather", total, size, t);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
