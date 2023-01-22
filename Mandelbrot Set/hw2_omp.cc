#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
	int chunk=500;
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
	
	/* MPI init */
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;
	int chunk = 500;
	MPI_Request req;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
	
    /* allocate memory for image */
	int local_height = (mpi_rank < height%mpi_ranks) ? height/mpi_ranks+1 : height/mpi_ranks;
	int* local_image = (int*)malloc(width * local_height * sizeof(int));
	assert(local_image);
	int start = (mpi_rank < height%mpi_ranks) ? (height/mpi_ranks+1)*mpi_rank : (height/mpi_ranks)*mpi_rank + height%mpi_ranks;
	
    /* mandelbrot set */
	#pragma omp parallel shared(local_image)
    {
		#pragma omp for schedule(dynamic,chunk) collapse(2) nowait
		for (int j = mpi_rank; j < height; j+=mpi_ranks) {
			for (int i = 0; i < width; ++i) {
				double y0 = j * ((upper - lower) / height) + lower;
				double x0 = i * ((right - left) / width) + left;
				int repeats = 0;
				double x = 0;
				double y = 0;
				double length_squared = 0;
				while (repeats < iters && length_squared < 4) {
					double temp = x * x - y * y + x0;
					y = 2 * x * y + y0;
					x = temp;
					length_squared = x * x + y * y;
					++repeats;
				}
				local_image[(j/mpi_ranks)*width+i] = repeats;
			}
		}
	}

	if (mpi_rank!=0){
		MPI_Isend(local_image, width*local_height, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, &req);
	}
	else{
		int* other_image = (int*)malloc(width * (height/mpi_ranks+1) * sizeof(int));
		int* global_image = (int*)malloc(width * height * sizeof(int));
		
		for(int h=0; h<height ; h+=mpi_ranks){
			for(int w=0; w<width; w++){
				global_image[h*width+w] = local_image[(h/mpi_ranks)*width+w];
			}
		}

		for(int other_rank=1; other_rank<mpi_ranks; other_rank++){
			int other_rank_height = (other_rank < height%mpi_ranks) ? height/mpi_ranks+1 : height/mpi_ranks;
			MPI_Irecv(other_image, width*other_rank_height, MPI_INT, other_rank, other_rank, MPI_COMM_WORLD, &req); //MPI_STATUS_IGNORE
			MPI_Wait(&req, MPI_STATUS_IGNORE);
			 
			for(int h=other_rank; h<height; h+=mpi_ranks){
				for(int w=0; w<width; w++){
					global_image[h*width+w] = other_image[(h/mpi_ranks)*width+w];
				}
			}
		}

		write_png(filename, iters, width, height, global_image);
		free(other_image);
		free(global_image);
	}
	

    /* draw and cleanup */
	free(local_image);
	MPI_Finalize();
}
