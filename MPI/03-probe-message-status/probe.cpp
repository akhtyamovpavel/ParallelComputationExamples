#include <mpi.h>
#include <iostream>
#include <ctime>
#include <cstdlib>


int main(int argc, char** argv) {
	
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	

	const int MAX_NUMBERS = 100;

	int number_amount = 0;

	if (world_rank == 0) {
		srand(time(NULL));
		int numbers[MAX_NUMBERS];
		number_amount = (rand() / float(RAND_MAX)) * MAX_NUMBERS;
		MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);

		std::cout << "0 sent " << number_amount << " numbers to 1" << std::endl;		
	} else if (world_rank == 1) {
		MPI_Status status;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

		MPI_Get_count(&status, MPI_INT, &number_amount);

		int* buffer = new int[number_amount];

		MPI_Recv(buffer, number_amount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		std::cout << "1 received " << number_amount << " numbers from 0. Message source is " << status.MPI_SOURCE << ", tag is " << status.MPI_TAG << std::endl;
	}

	
	MPI_Finalize();
	return 0;
}
