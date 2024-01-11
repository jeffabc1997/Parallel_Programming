#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_fp16.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define THREADNUMS 128
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

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int val[Z][3][THREADNUMS]; // lower precision and coalesced mem access
    __shared__ unsigned char r_mat[5][THREADNUMS+4]; // left boundary and right boundary + 2
    __shared__ unsigned char g_mat[5][THREADNUMS+4];
    __shared__ unsigned char b_mat[5][THREADNUMS+4];
    // no need to use smem for the mask
    if (tid >= width) return;

    int y = tid; // the variable means the y-axis value in an image
    int x = blockIdx.y; // use block y dimension to parallel x-axis in an image
    int thr_x = threadIdx.x;
    for (int v = -yBound; v <= yBound; ++v) {
        if(bound_check(x + v, 0, height)) {
            r_mat[v+yBound][thr_x + 2] = s[channels * (width * (x + v) + y) + 2]; // coalesced mem access
            g_mat[v+yBound][thr_x + 2] = s[channels * (width * (x + v) + y) + 1];
            b_mat[v+yBound][thr_x + 2] = s[channels * (width * (x + v) + y) + 0];
            if(thr_x <= 1) { // use 2 more threads to deal with the leftmost and rightmost pixels
                if(bound_check(y-2, 0, width)) {
                    r_mat[v+yBound][thr_x] = s[channels * (width * (x + v) + y-2) + 2];
                    g_mat[v+yBound][thr_x] = s[channels * (width * (x + v) + y-2) + 1];
                    b_mat[v+yBound][thr_x] = s[channels * (width * (x + v) + y-2) + 0];
                }
                // if(bound_check(y-1, 0, width)) {
                //     r_mat[v+yBound][thr_x + 1] = s[channels * (width * (x + v) + y-1) + 2];
                //     g_mat[v+yBound][thr_x + 1] = s[channels * (width * (x + v) + y-1) + 1];
                //     b_mat[v+yBound][thr_x + 1] = s[channels * (width * (x + v) + y-1) + 0];
                // }
            } else if(thr_x >= THREADNUMS-2) {
                if(bound_check(y+2, 0, width)) {
                    r_mat[v+yBound][thr_x + 4] = s[channels * (width * (x + v) + y+2) + 2];
                    g_mat[v+yBound][thr_x + 4] = s[channels * (width * (x + v) + y+2) + 1];
                    b_mat[v+yBound][thr_x + 4] = s[channels * (width * (x + v) + y+2) + 0];
                }
                // if(bound_check(y+1, 0, width)) {
                //     r_mat[v+yBound][thr_x + 3] = s[channels * (width * (x + v) + y+1) + 2];
                //     g_mat[v+yBound][thr_x + 3] = s[channels * (width * (x + v) + y+1) + 1];
                //     b_mat[v+yBound][thr_x + 3] = s[channels * (width * (x + v) + y+1) + 0];
                // }
            }
        }  
    }
    __syncthreads();
    
    /* Z axis of mask */
    for (int i = 0; i < Z; ++i) {
        val[i][2][threadIdx.x] = 0;
        val[i][1][threadIdx.x] = 0;
        val[i][0][threadIdx.x] = 0;
        /* Y and X axis of mask */
        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                if (bound_check(x + v, 0, height) && bound_check(y + u, 0, width)) {
                    val[i][2][threadIdx.x] += r_mat[v + yBound][thr_x + 2 + u] * mask[i][u + xBound][v + yBound]; // coalesced mem access for r_mat
                    val[i][1][threadIdx.x] += g_mat[v + yBound][thr_x + 2 + u] * mask[i][u + xBound][v + yBound];
                    val[i][0][threadIdx.x] += b_mat[v + yBound][thr_x + 2 + u] * mask[i][u + xBound][v + yBound];
                }
            }
        }
    }
    float totalR(0.); // lower precision
    float totalG(0.);
    float totalB(0.);
    for (int i = 0; i < Z; ++i) {
        totalR += (float)(val[i][2][threadIdx.x] * val[i][2][threadIdx.x]);
        totalG += (float)(val[i][1][threadIdx.x] * val[i][1][threadIdx.x]);
        totalB += (float)(val[i][0][threadIdx.x] * val[i][0][threadIdx.x]);
    }
    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = (totalR > 255.) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.) ? 255 : totalB;
    t[channels * (width * x + y) + 2] = cR;
    t[channels * (width * x + y) + 1] = cG;
    t[channels * (width * x + y) + 0] = cB;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault); // may be helpful

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // decide to use how many blocks and threads
    const int num_threads = THREADNUMS; // 128 seems ok
    const int num_x_blocks = width / num_threads + 1; 
    // dim3 block2D(16, 16)
    dim3 grid_size(num_x_blocks, height); // block (x, y) y controls a whole row with num_x_blocks
    // launch cuda kernel
    // std::cout << "height " << height << std::endl;
    sobel << <grid_size, num_threads>>> (dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

