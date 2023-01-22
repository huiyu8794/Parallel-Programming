#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm> 
#include <time.h>

#define TAG_COMPARE 0
#define TAG_SIZE 1

using namespace std;

int get_subarray_size(int rank, int num_processer, int num_element) { 
	int subarray_size = 0;
	if (num_processer <= num_element) {
		if (num_element % num_processer == 0) {
			subarray_size = num_element / num_processer; 
		} else {
			if(rank < num_element%num_processer) {
				subarray_size = num_element / num_processer + 1;
			} else {
				subarray_size = num_element / num_processer;
			}
		}
	} else {
		if(rank < num_element) {
			subarray_size = 1;
		} else {
			subarray_size = 0;
		}
	}

	return subarray_size;
}

int get_subarray_idx(int rank, int num_processer, int num_element, int subarray_size) {
	int index = 0;
	if(num_element % num_processer == 0 || num_element % num_processer > rank){
		index = rank * subarray_size;
	} else { 
		index = rank * subarray_size + num_element % num_processer;
	}

	return index;
}

void merge_and_sort(float arr_left[], float arr_right[], int m, int n, float new_data[], bool compare_small=false){
	
	if (compare_small==true){
		int i = 0, j = 0, k = 0; 
		while(i<m && j<n && k<m){
			if (arr_left[i] < arr_right[j]){
				new_data[k++] = arr_left[i++];
			}
			else{
				new_data[k++] = arr_right[j++];
			}
		}
		while(i<m && k<m){
			new_data[k++] = arr_left[i++];
		}
		while(j<n && k<m){
			new_data[k++] = arr_right[j++];
		}
	}
	else{
		int i=m-1, j=n-1 ,k=n-1;
		while(i>=0 && j>=0 && k>=0){
			if (arr_left[i] > arr_right[j]){
				new_data[k--] = arr_left[i--];
			}
			else{
				new_data[k--] = arr_right[j--];
			}
		}
		while(i>=0 && k>=0){
			new_data[k--] = arr_left[i--];
		}
		while(j>=0 && k>=0){
			new_data[k--] = arr_right[j--];
		}
	}
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_processer, subarray_idx, num_element, subarray_size;
	
	sscanf(argv[1], "%d", &num_element);
	
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processer);
	
    char *input_filename = argv[2];
    char *output_filename = argv[3];
	
    MPI_File input_file, output_file;
	MPI_Comm new_comm=MPI_COMM_WORLD;
	MPI_Group original_group, new_group;
	
	
	if (num_processer > num_element){
		int rank_list[num_element];
		
		for(int i=0; i<num_element; i++)
			rank_list[i]=i;

		MPI_Comm_group(MPI_COMM_WORLD, &original_group);
		MPI_Group_incl(original_group, num_element, rank_list, &new_group);
		MPI_Comm_create(new_comm, new_group, &new_comm);
		
		if (new_comm == MPI_COMM_NULL) {
			MPI_Finalize();
			exit(0);
		}
		num_processer = num_element;
	}

	subarray_size = get_subarray_size(rank, num_processer, num_element);
	float* data = (float*) malloc(subarray_size*sizeof(float));
	float* new_data = (float*) malloc((subarray_size+1) * sizeof(float));
	float* neighbor_subarray = (float*) malloc((subarray_size+1) * sizeof(float));

	subarray_idx = get_subarray_idx(rank, num_processer, num_element, subarray_size);
	
    MPI_File_open(new_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float)*subarray_idx, data, subarray_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
	
	sort(data, data+subarray_size);
	
	int still_exchange=1;
	int global_exchange=1;
	
	while(global_exchange){
		still_exchange=0;
		global_exchange=0;
		
		if (num_processer==1) break;
		
		
		if((num_processer%2==0) || (num_processer%2==1 && rank!=num_processer-1)){ // even phase
			if(rank%2==0){ // left item
				int right_subarray_size = 0;
				MPI_Send(&subarray_size, 1, MPI_INT, rank+1, TAG_SIZE, new_comm); // send the subarray size to right item
				MPI_Recv(&right_subarray_size, 1, MPI_INT, rank+1, TAG_SIZE, new_comm, MPI_STATUS_IGNORE); // recv the subarray size from right item

				MPI_Send(data, subarray_size, MPI_FLOAT, rank+1, TAG_COMPARE, new_comm); // send the whole subarray to right item
				MPI_Recv(neighbor_subarray, right_subarray_size, MPI_FLOAT, rank+1, TAG_COMPARE, new_comm, MPI_STATUS_IGNORE);  // recv the whole subarray from right item
				
				if(neighbor_subarray[0] < data[subarray_size-1]){ // check whether the two subarrays need to be sorted or not
					merge_and_sort(data, neighbor_subarray, subarray_size, right_subarray_size, new_data, true);
					memcpy(data, &new_data[0], subarray_size*sizeof(float));
					still_exchange=1;
				}
			}
			else if(rank%2==1){ // right item
				int left_subarray_size=0;
				MPI_Recv(&left_subarray_size, 1, MPI_INT, rank-1, TAG_SIZE, new_comm, MPI_STATUS_IGNORE); // recv the subarray size from left item
				MPI_Send(&subarray_size, 1, MPI_INT, rank-1, TAG_SIZE, new_comm); // send the subarray size to left item
				
				MPI_Recv(neighbor_subarray, left_subarray_size, MPI_FLOAT, rank-1, TAG_COMPARE, new_comm, MPI_STATUS_IGNORE); // recv the whole subarray from left item
				MPI_Send(data, subarray_size, MPI_FLOAT, rank-1, TAG_COMPARE, new_comm); // send the whole subarray to left item
				
				if(neighbor_subarray[left_subarray_size-1] > data[0]){ // check whether the two subarrays need to be sorted or not
					merge_and_sort(neighbor_subarray, data, left_subarray_size, subarray_size, new_data, false);
					memcpy(data, &new_data[0], subarray_size*sizeof(float));
					still_exchange=1;
				}
			}
		}
		
		
		if((num_processer%2==0 && rank!=0 && rank!=num_processer-1) || (num_processer%2==1 && rank!=0)){ // odd phase
			if(rank%2==1){ // left item
				int right_subarray_size = 0;
				MPI_Send(&subarray_size, 1, MPI_INT, rank+1, TAG_SIZE, new_comm);
				MPI_Recv(&right_subarray_size, 1, MPI_INT, rank+1, TAG_SIZE, new_comm, MPI_STATUS_IGNORE);

				MPI_Send(data, subarray_size, MPI_FLOAT, rank+1, TAG_COMPARE, new_comm);
				MPI_Recv(neighbor_subarray, right_subarray_size, MPI_FLOAT, rank+1, TAG_COMPARE, new_comm, MPI_STATUS_IGNORE);
				
				if(neighbor_subarray[0] < data[subarray_size-1]){
					merge_and_sort(data, neighbor_subarray, subarray_size, right_subarray_size, new_data, true);
					memcpy(data, &new_data[0], subarray_size*sizeof(float));
					still_exchange=1;
				}
			}
			else if(rank%2==0){ // right item
				int left_subarray_size=0;
				MPI_Recv(&left_subarray_size, 1, MPI_INT, rank-1, TAG_SIZE, new_comm, MPI_STATUS_IGNORE);
				MPI_Send(&subarray_size, 1, MPI_INT, rank-1, TAG_SIZE, new_comm);
				
				MPI_Recv(neighbor_subarray, left_subarray_size, MPI_FLOAT, rank-1, TAG_COMPARE, new_comm, MPI_STATUS_IGNORE);
				MPI_Send(data, subarray_size, MPI_FLOAT, rank-1, TAG_COMPARE, new_comm);
				
				if(neighbor_subarray[left_subarray_size-1] > data[0]){
					merge_and_sort(neighbor_subarray, data, left_subarray_size, subarray_size, new_data, false);
					memcpy(data, &new_data[0], subarray_size*sizeof(float));
					still_exchange=1;
				}
			}
		}
		
		MPI_Barrier(new_comm);
		MPI_Allreduce(&still_exchange, &global_exchange, 1, MPI_INT, MPI_SUM, new_comm);
	}
	
	free(neighbor_subarray);
	free(new_data);
    MPI_File_open(new_comm, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float)*subarray_idx, data, subarray_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

	
	free(data);
	
    MPI_Finalize();
    return 0;
}