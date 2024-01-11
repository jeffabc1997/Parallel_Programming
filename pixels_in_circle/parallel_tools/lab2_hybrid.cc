#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
using namespace std;
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	unsigned long long pixels = 0, y = 0;
	unsigned long long radius_start = r / mpi_ranks * mpi_rank;
	unsigned long long radius_end = (mpi_rank != mpi_ranks - 1)  ? r / mpi_ranks * (mpi_rank+1) : r;
	unsigned long long res = 0;
	
	#pragma omp parallel shared(pixels) firstprivate(y)
    {
		unsigned long long radius_sq = r * r;
		#pragma omp for schedule(static) nowait
		for(unsigned long long x = radius_start; x < radius_end; x++) {
			y += ceil(sqrtl(radius_sq - x*x));
			if(y & 256 == 1) {
				y %= k;
			}
			
		}
		#pragma omp critical
		{
			pixels += (y % k);
			pixels %= k;
			// cout << mpi_rank << ", " <<  pixels << ", " << endl;
		}
		
	}
	// Need Reduce here
	MPI_Reduce(&pixels, &res, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if(mpi_rank == 0) {
		printf("%llu\n", (4 * res) % k);
	}
    MPI_Finalize();
}
