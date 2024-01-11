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

#include <iostream>
#include <mpi.h>
#include <omp.h>

#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
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

/* use 2 __m128d to replace multiply twice in while loop */
/* with row_per_send = 2, force_even = 2, chunk = 1 (need to ensure all send is valid)*/
/* around 308 secs */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // double start_time = MPI_Wtime();
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

    int mpi_rank, mpi_nproc, mpi_master = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nproc);
    /* allocate memory for image */
    int* image = (mpi_rank == mpi_master) ? (int*)malloc(width * height * sizeof(int)) : NULL;
    if(mpi_rank == mpi_master) {
        assert(image);
    }
    int SIZE = 2;
    int row_per_send = 2; // determine how many rows we're gonna send once // 12 is better?
    int chunk = 1;
    int send_count = row_per_send * width; // master's view of send
    int recv_count = send_count;

    int* pixel_light = new int[recv_count];
    
    MPI_Status status; // use Non_blocking in the future
    
    if(mpi_rank == mpi_master) { /* master process */
        int row_send = 0, active_proc = 0;
        int terminate_tag = height;
        int row_write = 0;
    #pragma omp parallel shared(row_send, image)
    {
        if(omp_get_thread_num() == 0) { // master thread
            // Initail send
            for(int k = 1; k < mpi_nproc; k++) { // rank 1 ... n
                #pragma omp critical
                {
                    if(row_send + row_per_send <= height) {
                        MPI_Send(&send_count, 1, MPI_INT, k, row_send, MPI_COMM_WORLD);
                        row_send += row_per_send;
                        active_proc++;
                    } else {
                        MPI_Send(&send_count, 1, MPI_INT, k, terminate_tag, MPI_COMM_WORLD);
                    }
                }  
            }          
            int slave_available;
            bool terminate_process;
            while(active_proc > 0) {                  
                MPI_Recv(pixel_light, recv_count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                slave_available = status.MPI_SOURCE; // know who just sent us pixel_light
                row_write = status.MPI_TAG;
                active_proc--;
                if(row_write < height) { // need to write to the row which is in boundary
                    for(int j = 0; j < row_per_send; j++) { // j means which row
                        for(int i = 0; i < width; i++) { 
                            image[(j+row_write) * width + i] = pixel_light[i + j * width]; // receive and transmit to image; i want to use pixel_light[i+1] but fail
                        }
                    }
                }
                /* mechanism to prevent sending out of height */
                #pragma omp critical
                {
                    terminate_process = row_send + row_per_send > height;
                }
                /* send termination */
                if(terminate_process) {                    
                    MPI_Send(&send_count, 1, MPI_INT, slave_available, terminate_tag, MPI_COMM_WORLD);
                } else {
                    #pragma omp critical
                    {   
                        MPI_Send(&send_count, 1, MPI_INT, slave_available, row_send, MPI_COMM_WORLD);
                        row_send += row_per_send;
                    }
                    active_proc++;                     
                } 
            }
        } else { // master's slave threads
            int force_even = row_per_send, row_to_write; // we need even because it's easier for size-2 vector
            int i[SIZE]; // can be replace
            int j[SIZE];

            double length_squared[SIZE];
            double x0[SIZE];
            double y0[SIZE];
            bool write_already[SIZE];
            #pragma omp critical
            {   
                row_to_write = row_send;
                if(row_to_write + force_even <= height) {
                    row_send += force_even;
                }                 
            }
            int need_cal = force_even * width;
            while(row_to_write + force_even <= height) {
                #pragma omp parallel for firstprivate(i, j, x0, y0, length_squared, write_already) schedule(dynamic, 1)
                for(int t = 0; t < need_cal; t+=2) {   
                    for(int s = 0; s < SIZE; s++) {
                        i[s] = (t+s) % width; // x-axis
                        j[s] = (t+s) / width; // j[s] means which row of the data
                        
                        x0[s] = i[s] * ((right - left) / width) + left;
                        y0[s] = (j[s] + row_to_write) * ((upper - lower) / height) + lower;
                        write_already[s] = false; 
                    }
                    int break_var = 0, repeats = 0;
                    __m128d m_x0 = _mm_load_pd(x0);
                    __m128d m_y0 = _mm_load_pd(y0);
                    __m128d m_lsq; //= _mm_load_pd(length_squared);
                    __m128d m_x = _mm_set1_pd(0.0);
                    __m128d m_y = m_x;
                    __m128d temp;
                    __m128d m_xsq = m_x, m_ysq = m_x;
                    while (break_var < SIZE) { // we need to finish all the elements in a vector           
                        temp = _mm_add_pd(_mm_sub_pd(m_xsq, m_ysq), m_x0); // double temp = x * x - y * y + x0; // real number
                        m_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(m_x, m_y), _mm_set1_pd(2.0)), m_y0); // y = 2 * x * y + y0; // imagine number
                        m_x = temp; // x = temp;
                        m_xsq = _mm_mul_pd(m_x, m_x);
                        m_ysq = _mm_mul_pd(m_y, m_y);
                        m_lsq = _mm_add_pd(m_xsq, m_ysq); // length_squared = x * x + y * y; // lose 0.01% correctness
                        _mm_store_pd(length_squared, m_lsq); // can we fix this to "not store, and compare in for loop"?

                        repeats++;
                        for(int s = 0; s < SIZE; s++) { // here we need to fix it to __m, or we can use lower half only?
                            if((write_already[s] == false) && ((length_squared[s] > 4) || (repeats == iters))) { 
                                image[(j[s] + row_to_write) * width + i[s]] = repeats;
                                write_already[s] = true;
                                break_var++;
                            } 
                        }   
                    }
                }
                #pragma omp critical
                {   
                    row_to_write = row_send;
                    if(row_send + force_even <= height) {
                        row_send += force_even;
                    }                 
                }
            }
        }
        
    }       
        /* master process deals with the leftover pixels */
    if(row_send < height) {
        
        double r_l_w = ((right - left) / width);
        double u_l_h = ((upper - lower) / height);
        int i[SIZE]; // can be replace
        int j[SIZE];
        double x0[SIZE];
        double y0[SIZE];
        double length_squared[SIZE];

        bool write_already[SIZE];
        int ttl_pixel = (height - row_send) * width;
        int need_cal = (ttl_pixel >> 1) << 1;
        
        #pragma omp parallel for firstprivate(i, j, x0, y0, length_squared, write_already) schedule(dynamic, chunk)
        for(int t = 0; t < need_cal; t+=2) {
            for(int s = 0; s < SIZE; s++) {
                i[s] = (t+s) % width; // x-axis
                j[s] = (t+s) / width; // j[s] means which row of the data
                
                x0[s] = i[s] * r_l_w + left;
                y0[s] = (j[s] + row_send) * u_l_h + lower;
                write_already[s] = false; 
            }
            int break_var = 0, repeats = 0;
            __m128d m_x0 = _mm_load_pd(x0);
            __m128d m_y0 = _mm_load_pd(y0);
            __m128d m_lsq; //= _mm_load_pd(length_squared);
            __m128d m_x = _mm_set1_pd(0.0);
            __m128d m_y = _mm_set1_pd(0.0);
            __m128d temp;
            __m128d m_xsq = _mm_set1_pd(0.0), m_ysq = _mm_set1_pd(0.0);
            while (break_var < SIZE) { // we need to finish all the elements in a vector           
                temp = _mm_add_pd(_mm_sub_pd(m_xsq, m_ysq), m_x0); // double temp = x * x - y * y + x0; // real number
                m_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(m_x, m_y), _mm_set1_pd(2.0)), m_y0); // y = 2 * x * y + y0; // imagine number
                m_x = temp; // x = temp;
                m_xsq = _mm_mul_pd(m_x, m_x);
                m_ysq = _mm_mul_pd(m_y, m_y);
                m_lsq = _mm_add_pd(m_xsq, m_ysq); // length_squared = x * x + y * y; // lose 0.01% correctness
                
                _mm_store_pd(length_squared, m_lsq); // can we fix this to "not store, and compare in for loop"?

                repeats++;
                for(int s = 0; s < SIZE; s++) { // here we need to fix it to __m, or we can use lower half only?
                    if((write_already[s] == false) && ((length_squared[s] > 4) || (repeats == iters))) { 
                        image[(j[s] + row_send)* width + i[s]] = repeats;
                        write_already[s] = true;
                        break_var++;
                    } 
                }   
            } 
        }

        if((ttl_pixel & 1) == 1) { // deal with the last pixel
            double x = 0;
            double y = 0;
            i[0] = width - 1;
            j[0] = height - 1;
            x0[0] = i[0] * ((right - left) / width) + left;
            y0[0] = (j[0]) * ((upper - lower) / height) + lower;

            length_squared[0] = 0;
            int repeats = 0;
            double temp;
            while (repeats < iters && length_squared[0] < 4.0) {
                temp = x * x - y * y + x0[0]; // real number
                y = 2 * x * y + y0[0]; // imagine number
                x = temp;
                length_squared[0] = x * x + y * y;
                ++repeats;
            }
            image[width*height-1] = repeats;
        }
    }
    
    write_png(filename, iters, width, height, image);
    if(image) {
        free(image); 
        image = NULL;
    }
    
    } else { /* slave process */         
        MPI_Recv(&send_count, 1, MPI_INT, mpi_master, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // get source_tag from master's data_tag
 
        int row_write = status.MPI_TAG;
        int ttl_pixel = row_per_send * width;

        double r_l_w = ((right - left) / width);
        double u_l_h = ((upper - lower) / height);
        int i[SIZE]; 
        int j[SIZE];
        double x0[SIZE];
        double y0[SIZE];
        double length_squared[SIZE];
        double x[SIZE];
        double y[SIZE];
        bool write_already[SIZE];

        while(row_write < height) { // use TAG to control termination
            int need_cal = (ttl_pixel >> 1) << 1;
            #pragma omp parallel for firstprivate(i, j, x0, y0, length_squared, x, y, write_already) schedule(dynamic, chunk)
                for(int t = 0; t < need_cal; t+=2) {
                    for(int s = 0; s < SIZE; s++) {
                        i[s] = (t+s) % width; // x-axis
                        j[s] = (t+s) / width; // j[s] means which row of the data
                        
                        x0[s] = i[s] * r_l_w + left;
                        y0[s] = (j[s] + row_write) * u_l_h + lower;

                        write_already[s] = false; 
                    }
                    int break_var = 0, repeats = 0;
                    __m128d m_x0 = _mm_load_pd(x0);
                    __m128d m_y0 = _mm_load_pd(y0);
                    __m128d m_lsq; //= _mm_load_pd(length_squared);
                    __m128d m_x = _mm_set1_pd(0.0);
                    __m128d m_y = _mm_set1_pd(0.0);
                    __m128d temp;
                    __m128d m_xsq = _mm_set1_pd(0.0), m_ysq = _mm_set1_pd(0.0);
                    while (break_var < SIZE) { // we need to finish all the elements in a vector           
                        temp = _mm_add_pd(_mm_sub_pd(m_xsq, m_ysq), m_x0); // double temp = x * x - y * y + x0; // real number
                        m_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(m_x, m_y), _mm_set1_pd(2.0)), m_y0); // y = 2 * x * y + y0; // imagine number
                        m_x = temp; // x = temp;
                        m_xsq = _mm_mul_pd(m_x, m_x);
                        m_ysq = _mm_mul_pd(m_y, m_y);
                        m_lsq = _mm_add_pd(m_xsq, m_ysq); 
                        _mm_store_pd(length_squared, m_lsq); // can we fix this to "not store, and compare in for loop"?

                        repeats++;
                        for(int s = 0; s < SIZE; s++) { // here we need to fix it to __m, or we can use lower half only?
                            if((write_already[s] == false) && ((length_squared[s] > 4) || (repeats == iters))) { 
                                pixel_light[j[s] * width + i[s]] = repeats;
                                write_already[s] = true;
                                break_var++;
                            } 
                        }   
                    } 
                }
            if((ttl_pixel & 1) == 1) { // deal with the last pixel
                i[0] = width - 1;
                j[0] = row_per_send - 1;
                x0[0] = i[0] * ((right - left) / width) + left;
                y0[0] = (j[0] + row_write) * ((upper - lower) / height) + lower;
                x[0] = 0;
                y[0] = 0;
                length_squared[0] = 0;
                int repeats = 0;
                double temp;
                while (repeats < iters && length_squared[0] < 4.0) {
                    temp = x[0] * x[0] - y[0] * y[0] + x0[0]; // real number
                    y[0] = 2 * x[0] * y[0] + y0[0]; // imagine number
                    x[0] = temp;
                    length_squared[0] = x[0] * x[0] + y[0] * y[0];
                    ++repeats;
                }
                pixel_light[ttl_pixel-1] = repeats;
            }
            MPI_Sendrecv(pixel_light, recv_count, MPI_INT, mpi_master, row_write, &send_count, 1, MPI_INT, mpi_master, MPI_ANY_TAG, 
                            MPI_COMM_WORLD, &status);
            row_write = status.MPI_TAG;
        }
    }
    
    delete []pixel_light;
    MPI_Finalize();
    /* draw and cleanup */

}
