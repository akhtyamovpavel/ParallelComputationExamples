#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>
#include <unistd.h>

int main(int argc, char** argv) {

	int num_threads = omp_get_num_threads();

	int threads_counter[10];
	for (int index = 0; index < 10; ++index) {
		threads_counter[index] = 0;
	}

	int thread_id;

#pragma omp parallel private(num_threads, thread_id)
	{
		#pragma omp critical(A)
		{
			thread_id = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			sleep(1);
			std::cout << num_threads << " " << thread_id << std::endl;
		}

		#pragma omp critical(B)
		{
			thread_id = omp_get_thread_num();
			num_threads = omp_get_num_threads();
			sleep(1);
			std::cout << num_threads << " " << thread_id << std::endl;
		}
		
		thread_id = omp_get_thread_num();
		std::cout << thread_id << std::endl;

	}

	return 0;
}
