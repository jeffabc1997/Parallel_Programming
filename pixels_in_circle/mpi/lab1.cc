#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long radius = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long result = 0;
    unsigned long long square_radius = radius * radius;
	int numtasks, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	for(unsigned long long int r = rank; r < radius; r += numtasks) {
		unsigned long long y = ceil(sqrtl(square_radius - r * r));
		pixels += y;
		pixels %= k;
	}
	// printf("Hello. I am %d of %d. I have %d pixels\n", rank, numtasks, pixels);
	
	/* should Add up all results here */
	MPI_Reduce(&pixels, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
    MPI_Finalize();
	if(rank == 0) {
		printf("%llu\n", (4 * result) % k); // mutiply by 4
	}
    return 0;
} 