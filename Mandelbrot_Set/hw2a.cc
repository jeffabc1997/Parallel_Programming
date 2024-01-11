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

#include <pthread.h>
#include <iostream>

#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

int I0 = 0;
int J0 = 0;
int Normal_size = 16; // we can determine how big will the batch be.
int SIZE = 2; // 128 bits for a vector with 2 double, so SIZE = 2
pthread_mutex_t mutex;
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

class myParam {
public:
    myParam() {}
    myParam(int it, double l, double r, double low, double up, int wid, int ht, int* img, unsigned long long id); 
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    int* image;
    unsigned long long thread_id;
};

/* calculate the value of pixels on the image */
void* calculate(void* param) {
    myParam* args = (myParam*) param;
    int pixel_i[Normal_size]; // we take a batch of i and j during 1 mutex lock
    int pixel_j[Normal_size];
    int ttl_pixel;
    int need_cal = 0; // if the I0 and J0 go to the end, then need_cal can detect, not sure if we really need this
    pthread_mutex_lock(&mutex);
    for(int t = 0; t < Normal_size; t++) {
        pixel_i[t] = I0++;
        pixel_j[t] = J0;
        need_cal++;
        if(I0 >= args->width) { // if I0 out of x-axis then we move J0 to upper y-axis
            J0 += 1;  
            I0 = 0;
        }
        if(J0 > args->height) { // y-axis boundary as termination
            break;
        }
    }
    pthread_mutex_unlock(&mutex);
    double x0[SIZE];
    double y0[SIZE];
    double length_squared[SIZE];
    bool write_already[SIZE]; // if that (i, j) has repeats value, then it's done
    while(pixel_j[need_cal-1] < args->height) {
        ttl_pixel = need_cal;
        need_cal = (need_cal >> 1) << 1;
        /* mandelbrot set */
        #pragma GCC ivdep
        for(int t = 0; t < need_cal; t+=2) { // need_cal is the number of elements we get from one lock
            // Initialize
            #pragma GCC ivdep
            for(int s = 0; s < SIZE; s++) {
                y0[s] = pixel_j[s+t] * ((args->upper - args->lower) / args->height) + args->lower;
                x0[s] = pixel_i[s+t] * ((args->right - args->left) / args->width) + args->left;
                write_already[s] = false; 
            }
            int break_var = 0;
            int repeats = 0;
            
            // load vector
            __m128d m_x0 = _mm_load_pd(x0);
            __m128d m_y0 = _mm_load_pd(y0);
            __m128d m_lsq; // = _mm_load_pd(length_squared);
            __m128d m_x = _mm_set1_pd(0.0);
            __m128d m_y = _mm_set1_pd(0.0);
            __m128d temp;
            while (break_var < SIZE) { // we need to finish all the elements in a vector           
                temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(m_x, m_x), _mm_mul_pd(m_y, m_y)), m_x0); // double temp = x * x - y * y + x0; // real number
                m_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(m_x, m_y), _mm_set1_pd(2.0)), m_y0); // y = 2 * x * y + y0; // imagine number
                m_x = temp; // x = temp;
                m_lsq = _mm_add_pd(_mm_mul_pd(m_x, m_x), _mm_mul_pd(m_y, m_y)); // length_squared = x * x + y * y;
                _mm_store_pd(length_squared, m_lsq); // can we fix this to "not store, and compare in for loop"?

                repeats++;
                for(int s = 0; s < SIZE; s++) { // here we need to fix it to __m, or we can use lower half only?
                    if((write_already[s] == false) && ((length_squared[s] > 4) || (repeats == args->iters))) { 
                        args->image[pixel_j[s+t] * args->width + pixel_i[s+t]] = repeats;
                        write_already[s] = true;
                        break_var++;
                    } 
                }   
            }
        }
        if((ttl_pixel & 1) == 1) { // deal with the last pixel
            y0[0] = pixel_j[ttl_pixel-1] * ((args->upper - args->lower) / args->height) + args->lower;
            x0[0] = pixel_i[ttl_pixel-1] * ((args->right - args->left) / args->width) + args->left;
            double x = 0;
            double y = 0;
            length_squared[0] = 0;
            int repeats = 0;
            while (repeats < args->iters && length_squared[0] < 4.0) {
                double temp = x * x - y * y + x0[0]; // real number
                y = 2 * x * y + y0[0]; // imagine number
                x = temp;
                length_squared[0] = x * x + y * y;
                ++repeats;
            }
            args->image[pixel_j[ttl_pixel-1]*args->width + pixel_i[ttl_pixel-1]] = repeats;
        }
        need_cal = 0;
        pthread_mutex_lock(&mutex);
        for(int t = 0; t < Normal_size; t++) {
            pixel_i[t] = I0++;
            pixel_j[t] = J0;
            need_cal++;
            if(I0 >= args->width) {
                J0 += 1;
                I0 = 0;             
            }
            if(J0 > args->height) {
                break;
            }
        }
        pthread_mutex_unlock(&mutex); 
    }
    pthread_exit(NULL);
}
/* little optimization from ver2, and get rid of the function with flattened code */
/* Normal_size = 16, 432.16 secs with flag -fno-signed-zeros */
int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

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
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    memset(image, 0, width * height);
    unsigned long long ncpus = CPU_COUNT(&cpu_set);
    pthread_t threads[ncpus];
    int rc; 
    
    pthread_mutex_init(&mutex, NULL);
    myParam* param = new myParam[ncpus];
    for (unsigned long long i = 0; i < ncpus; i++) {
        param[i] = myParam(iters, left, right, lower, upper, width, height, image, i);
        rc = pthread_create(&threads[i], NULL, calculate, (void*)&param[i]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    for(unsigned long long i = 0; i < ncpus; i++) {
		pthread_join(threads[i], NULL);
	}
    

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    delete []param;
    free(image);
    pthread_mutex_destroy(&mutex);
}


myParam::myParam(int it, double l, double r, double low, double up, int wid, int ht, int* img, unsigned long long id) {
    iters = it;
    left = l;
    right = r;
    lower = low;
    upper = up;
    width = wid;
    height = ht;
    image = img;
    thread_id = id;
}