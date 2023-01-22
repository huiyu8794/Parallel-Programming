#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {

  	MPI_Init(&argc, &argv);

  	//double starttime, endtime;
	//starttime = MPI_Wtime();

  	int rank = 0, size = 0;

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

  	MPI_Comm_size(MPI_COMM_WORLD, &size);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
  	unsigned long long this_rank = (unsigned long long)rank;
	unsigned long long long_size = (unsigned long long)size;

	unsigned long long localSum = 0, globalSum = 0;
	while(this_rank < r) {
    	localSum += ((unsigned long long)(ceil(sqrtl((unsigned long long)(r*r - this_rank*this_rank)))));
		localSum %= k;
    	this_rank+=long_size;
	}

  	MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
 	globalSum=(globalSum*4)%k;

	//endtime = MPI_Wtime();
   
  	//printf("That took %f seconds\n",endtime-starttime);
			
   if(rank==0){
     printf("%llu\n", globalSum);
   }
   
   MPI_Finalize();
  	
  	return 0;
}
