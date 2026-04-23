#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Декартова топология: MPI_Cart_create оборачивает коммуникатор в 2D/3D-...
// решётку. Для каждого ранга доступны его координаты (MPI_Cart_coords)
// и соседи по каждому измерению (MPI_Cart_shift) — именно то, что
// нужно для stencil-вычислений (обновление точки через "верх/низ/лево/право").
//
// Здесь строим 2D-решётку. Размеры решётки MPI подбирает сам через
// MPI_Dims_create. На краях, где соседа нет, MPI_Cart_shift возвращает
// MPI_PROC_NULL — это корректный "пустой" пир: MPI_Send/Recv с ним
// просто не выполняют обмен, что удобно для однородной записи кода.

static void print_ordered(MPI_Comm comm, const std::string& line) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0) {
        std::cout << line << std::endl;
        for (int r = 1; r < size; ++r) {
            MPI_Status st;
            MPI_Probe(r, 0, comm, &st);
            int nbytes = 0;
            MPI_Get_count(&st, MPI_CHAR, &nbytes);
            std::vector<char> buf(nbytes);
            MPI_Recv(buf.data(), nbytes, MPI_CHAR, r, 0, comm, MPI_STATUS_IGNORE);
            std::cout << std::string(buf.begin(), buf.end()) << std::endl;
        }
    } else {
        MPI_Send(line.data(), static_cast<int>(line.size()), MPI_CHAR,
                 0, 0, comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // dims = {0, 0} → MPI_Dims_create сам подберёт "квадратное" разбиение.
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);

    int periods[2] = {0, 0};   // без периодичности (края "обрезаны")
    int reorder = 0;           // сохранить исходные ранги

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);

    if (world_rank == 0) {
        std::cout << "=== MPI_Cart_create: 2D-решётка "
                  << dims[0] << "x" << dims[1]
                  << ", world_size=" << world_size << " ===" << std::endl;
    }

    int coords[2] = {0, 0};
    MPI_Cart_coords(cart, world_rank, 2, coords);

    int up = MPI_PROC_NULL;
    int down = MPI_PROC_NULL;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;
    // MPI_Cart_shift(comm, dim, disp, &src, &dst):
    //   disp=+1 — сосед "со следующим индексом" по dim (он же dst);
    //   src — кто бы прислал нам по тому же правилу (т.е. предыдущий сосед).
    // Для dim=0 (строки): src=up,  dst=down.
    // Для dim=1 (столбцы): src=left, dst=right.
    MPI_Cart_shift(cart, 0, 1, &up,   &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    std::ostringstream oss;
    oss << "  rank=" << world_rank
        << "  coords=(" << coords[0] << "," << coords[1] << ")"
        << "  up="    << up
        << "  down="  << down
        << "  left="  << left
        << "  right=" << right;
    print_ordered(MPI_COMM_WORLD, oss.str());

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
