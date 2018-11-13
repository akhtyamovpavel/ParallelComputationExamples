#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>


int main(int argc, char** argv) {

	int num_threads = omp_get_num_threads();
	std::cout << num_threads << std::endl;

	int threads_counter[10];
	for (int index = 0; index < 10; ++index) {
		threads_counter[index] = 0;
	}

#pragma omp parallel private(num_threads)
	{
		#pragma omp master
		{
			num_threads = omp_get_num_threads();
			std::cout << num_threads << std::endl;
		}

	}

	return 0;
}
