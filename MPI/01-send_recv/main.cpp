#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int number;

    int array[10];

    MPI_Status status;
    if (world_rank == 0) {
        for (int i = 0; i < 10; ++i) {
            array[i] = i;
        }

        MPI_Send(
            &array[5] /* pointer to start */,
            5 /* number of words */,
            MPI_INT /* word type */, 
            1 /* rank of receiver */, 
            0 /* tag */,
            MPI_COMM_WORLD /* communicator: default - all world */
        );
    } else if (world_rank == 1) {
        MPI_Recv(
            &array[5] /* pointer to start */, 
            5 /* number of words */, 
            MPI_INT /* type of word */, 
            0 /* rank of sender */, 
            0 /* tag */, 
            MPI_COMM_WORLD /* communicator: default - all world */,
            &status
        );
        std::cout << "Process 1 received 5 elements from process 0. They are" << std::endl;

        for (int i = 5; i < 10; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

    
    MPI_Finalize();
    return 0;
}
