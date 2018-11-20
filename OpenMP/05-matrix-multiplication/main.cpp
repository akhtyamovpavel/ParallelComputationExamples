#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>


int main(int argc, char** argv) {
	int num_elements = atoi(argv[1]);
	long long **a = new long long*[num_elements];
	long long **b = new long long*[num_elements];

	long long **result = new long long*[num_elements];

	for (int index = 0; index < num_elements; ++index) {
		a[index] = new long long[num_elements];
		b[index] = new long long[num_elements];
		result[index] = new long long [num_elements];
	}

	for (int index = 0; index < num_elements; ++index) {
		for (int jndex = 0; jndex < num_elements; ++jndex) {
			a[index][jndex] = index;
			b[index][jndex] = jndex;

			result[index][jndex] = 0;
		}
	}



	int parallel_index;
	int parallel_jndex;

	double start_time = omp_get_wtime();

#pragma omp parallel for default(shared) private(parallel_index, parallel_jndex)
	for (parallel_jndex = 0; parallel_jndex < num_elements; ++parallel_jndex) {
		for (parallel_index = 0; parallel_index < num_elements; ++parallel_index) {
			for (int kndex = 0; kndex < num_elements; ++kndex) {
				result[parallel_index][parallel_jndex] = result[parallel_index][parallel_jndex] + a[parallel_index][kndex] * b[kndex][parallel_jndex];
			}
		}
	}

	double end_time = omp_get_wtime();

	std::cout << "Time: " << end_time - start_time << std::endl;

	for (int index = 0; index < num_elements; ++index) {
		delete[] result[index];
		delete[] a[index];
		delete[] b[index];
	}
	delete[] a;
	delete[] b;
	delete[] result;
	return 0;
}
