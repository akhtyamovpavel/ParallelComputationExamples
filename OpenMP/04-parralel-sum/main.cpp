#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>


int main(int argc, char** argv) {
	int num_elements = atoi(argv[1]);
	long long *a = new long long[num_elements];
	long long *b = new long long[num_elements];

	long long result = 0;
	for (int index = 0; index < num_elements; ++index) {
		a[index] = index;
		b[index] = 2 * index;
	}

	int parallel_index;


	double start_time = omp_get_wtime();

#pragma omp parallel default(shared) private(parallel_index)
	{
		#pragma omp for schedule(static, 10000) reduction(+:result)
		for (parallel_index = 0; parallel_index < num_elements; ++parallel_index) {
			result = result + a[parallel_index] + b[parallel_index];			
		}
	}

	double end_time = omp_get_wtime();

	std::cout << "Time: " << end_time - start_time << std::endl;

	std::cout << result << std::endl;
	delete[] a;
	delete[] b;
	return 0;
}
