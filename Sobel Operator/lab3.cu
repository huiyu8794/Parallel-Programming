#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
// #include <stdlib.h>
// #include <assert.h>
// #include <math.h>
// #include <utils.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

/* Hint 7 */
// device side can't call host function
// declare it to device function
__device__  inline int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}


int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel_kernel(unsigned char *s, unsigned char *t, unsigned *height, unsigned *width,
           unsigned *channels) {
    /* Hint 4 */
    // get tid by blockIdx, blockDim threadIdx 
    // and replace y or x by tid
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= *height){
        // printf("tid %4d return\n", tid); 
        return;
    }

    /* Hint 5 */
    // use constant memory or shared memory for filter matrix
    // sync threads if necessary
    const char filter[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                 { -2, -8, -12, -8, -2 },
                                 { 0, 0, 0, 0, 0 },
                                 { 2, 8, 12, 8, 2 },
                                 { 1, 4, 6, 4, 1 } },
                               { { -1, -2, 0, 2, 1 },
                                 { -4, -8, 0, 8, 4 },
                                 { -6, -12, 0, 12, 6 },
                                 { -4, -8, 0, 8, 4 },
                                 { -1, -2, 0, 2, 1 } } };
    double val[Z][3];
    int y = tid;
    // for (int y = tid; y < *height; y += gridDim.x * blockDim.x) {
        // printf("Thread %d, y = %4d, height = %4d\n", tid, y, *height);
        for (int x = 0; x < *width; ++x) {
            /* Z axis of filter */
            // printf("Thread %d, y = %4d, x = %4d\n", tid, y, x);
            for (int i = 0; i < Z; ++i) {
                
                val[i][2] = 0.;
                val[i][1] = 0.;
                val[i][0] = 0.;

                /* Y and X axis of filter */
                for (int v = -yBound; v <= yBound; ++v) {
                    for (int u = -xBound; u <= xBound; ++u) {
                        if (bound_check(x + u, 0, *width) && bound_check(y + v, 0, *height)) {
                            const unsigned char R =
                                s[*channels * (*width * (y + v) + (x + u)) + 2];
                            const unsigned char G =
                                s[*channels * (*width * (y + v) + (x + u)) + 1];
                            const unsigned char B =
                                s[*channels * (*width * (y + v) + (x + u)) + 0];
                            val[i][2] += R * filter[i][u + xBound][v + yBound];
                            val[i][1] += G * filter[i][u + xBound][v + yBound];
                            val[i][0] += B * filter[i][u + xBound][v + yBound]; 
                        }
                    }
                }
            }
            // printf("tid = %d Loop finished\n", tid);
            double totalR = 0.;
            double totalG = 0.;
            double totalB = 0.;
            for (int i = 0; i < Z; ++i) {
                totalR += val[i][2] * val[i][2];
                totalG += val[i][1] * val[i][1];
                totalB += val[i][0] * val[i][0];
            }
            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.) ? 255 : totalB;
            t[*channels * (*width * y + x) + 2] = cR;
            t[*channels * (*width * y + x) + 1] = cG;
            t[*channels * (*width * y + x) + 0] = cB;
        }
    // }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned *height_cuda, *width_cuda, *channels_cuda;
    unsigned char *src = NULL, *dst;
    unsigned char *src_cuda = NULL, *dst_cuda;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels *
                                  sizeof(unsigned char));
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    cudaMalloc((void **)&src_cuda, height * width * channels *sizeof(unsigned char));
    cudaMalloc((void **)&dst_cuda, height * width * channels *sizeof(unsigned char));
    cudaMalloc((void **)&height_cuda, sizeof(unsigned));
    cudaMalloc((void **)&width_cuda, sizeof(unsigned));
    cudaMalloc((void **)&channels_cuda, sizeof(unsigned));
    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(src_cuda, src, height * width * channels *sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_cuda, dst, height * width * channels *sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(height_cuda, &height, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(width_cuda, &width, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(channels_cuda, &channels, sizeof(unsigned), cudaMemcpyHostToDevice);;
    /* Hint 3 */
    // decide to use how many blocks and threads
    int threads_per_block = 100;
    sobel_kernel<<<height/threads_per_block + 1,threads_per_block>>>(src_cuda, dst_cuda, height_cuda, width_cuda, channels_cuda);
    // launch cuda kernel

    /* computation */
    
    // sobel(src, dst, height, width, channels);

    /* Hint 6 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, dst_cuda, height * width * channels *sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(src_cuda); cudaFree(dst_cuda); cudaFree(height_cuda); cudaFree(width_cuda); cudaFree(channels_cuda);
    write_png(argv[2], dst, height, width, channels);
    return 0;
}
