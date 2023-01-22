#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks, omp_threads, omp_thread;
	
	int chunk = 100;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	
	unsigned long long pixels = 0;
	unsigned long long total = 0;
	unsigned long long iter = (mpi_rank==mpi_ranks-1) ? (unsigned long long) r/mpi_ranks + r%mpi_ranks : (unsigned long long) r/mpi_ranks;
	unsigned long long offset = (r/mpi_ranks)*mpi_rank;
	
    omp_lock_t lock;
    omp_init_lock(&lock);
	#pragma omp parallel shared(pixels)
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
		
		#pragma omp for schedule(dynamic,chunk) reduction(+:pixels) ordered 
		for(unsigned long long i=offset;i<offset+iter;i++){
			unsigned long long y = (unsigned long long) ceil(sqrtl(r*r - i*i));
			pixels += y;
		}
		
	}
	omp_destroy_lock(&lock);
	
	pixels %= k;
	MPI_Reduce(&pixels, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (mpi_rank == 0) printf("%llu\n", (4 * total) % k);
	
	MPI_Finalize();
	
    return 0;
}
