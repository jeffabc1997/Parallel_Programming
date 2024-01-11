#include <stdio.h>
#include <stdlib.h>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#define BLOCKFACTOR 64
#define WARPSIZE BLOCKFACTOR/2
using namespace std;
const int INF = ((1 << 30) - 1);

void input(char* inFileName);
void output(char* outFileName);

void block_FW();
int n, m, Round_n;
int* dst;
__device__ int myMin(int original, int sum_dst) {
    return sum_dst * (sum_dst < original) + original * (sum_dst >= original);
}
/* coalesce done 32x32 */
__global__ void test_phase1(int Round, int* dst, int graph_len) {
    __shared__ int s_ij[BLOCKFACTOR][BLOCKFACTOR];
    int thr_x = threadIdx.x;
    int thr_y = threadIdx.y;
    
    int k = Round * BLOCKFACTOR; // start index of the real matrix
    int x_offset = thr_x + WARPSIZE;
    int y_offset = thr_y + WARPSIZE;
    s_ij[thr_y][thr_x] = dst[(k + thr_y) * graph_len + (k + thr_x)]; // thread(1,0) moves data to s[0, 1] // left upper vertex
    s_ij[thr_y][x_offset] = dst[(k + thr_y) * graph_len + (k + x_offset)]; // right upper of the whole block
    s_ij[y_offset][thr_x] = dst[(k + y_offset) * graph_len + (k + thr_x)]; // left bottom
    s_ij[y_offset][x_offset] = dst[(k + y_offset) * graph_len + (k + x_offset)]; // right bottom
    __syncthreads();
    
    // #pragma unroll 64
    for (int p=0; p < BLOCKFACTOR; ++p) { // need to make p fit in
        s_ij[thr_y][thr_x] = min(s_ij[thr_y][thr_x], s_ij[thr_y][p] + s_ij[p][thr_x]);
        s_ij[thr_y][x_offset] = min(s_ij[thr_y][x_offset], s_ij[thr_y][p] + s_ij[p][x_offset]);
        s_ij[y_offset][thr_x] = min(s_ij[y_offset][thr_x], s_ij[y_offset][p] + s_ij[p][thr_x]);
        s_ij[y_offset][x_offset] = min(s_ij[y_offset][x_offset], s_ij[y_offset][p] + s_ij[p][x_offset]);         
        __syncthreads();
    }
    dst[(k + thr_y) * graph_len + (k + thr_x)] = s_ij[thr_y][thr_x];
    dst[(k + thr_y) * graph_len + (k + x_offset)] = s_ij[thr_y][x_offset];
    dst[(k + y_offset) * graph_len + (k + thr_x)] = s_ij[y_offset][thr_x];
    dst[(k + y_offset) * graph_len + (k + x_offset)] = s_ij[y_offset][x_offset];
}

__global__ void phase2_combine(int Round, int* dst, int graph_len) {
    __shared__ int s_core[BLOCKFACTOR][BLOCKFACTOR];
    int k = Round * BLOCKFACTOR;
    int thr_x = threadIdx.x;
    int thr_y = threadIdx.y;
    int x_offset = thr_x + WARPSIZE;
    int y_offset = thr_y + WARPSIZE;
    if(blockIdx.y == 0) {
        
        __shared__ int s_ik[BLOCKFACTOR][BLOCKFACTOR];
        s_core[thr_y][thr_x] = dst[(k + thr_y) * graph_len + (k + thr_x)]; // the 64 * 64 matrix core
        s_core[thr_y][x_offset] = dst[(k + thr_y) * graph_len + (k + x_offset)];
        s_core[y_offset][thr_x] = dst[(k + y_offset) * graph_len + (k + thr_x)];
        s_core[y_offset][x_offset] = dst[(k + y_offset) * graph_len + (k + x_offset)];

        s_ik[thr_y][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + thr_x)];
        s_ik[thr_y][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + x_offset)];
        s_ik[y_offset][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + thr_x)];
        s_ik[y_offset][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + x_offset)];
        __syncthreads();
        for (int p=0; p < BLOCKFACTOR; ++p) { // need to make p fit in
            s_ik[thr_y][thr_x] = myMin(s_ik[thr_y][thr_x], s_ik[thr_y][p]+ s_core[p][thr_x]);
            s_ik[thr_y][x_offset] = myMin(s_ik[thr_y][x_offset], s_ik[thr_y][p] + s_core[p][x_offset]);
            s_ik[y_offset][thr_x] = myMin(s_ik[y_offset][thr_x], s_ik[y_offset][p] + s_core[p][thr_x]);
            s_ik[y_offset][x_offset] = myMin(s_ik[y_offset][x_offset], s_ik[y_offset][p] + s_core[p][x_offset]);
        } 
        dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + thr_x)] = s_ik[thr_y][thr_x];
        dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + x_offset)] = s_ik[thr_y][x_offset];
        dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + thr_x)] = s_ik[y_offset][thr_x];
        dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + x_offset)] = s_ik[y_offset][x_offset];
    } else if (blockIdx.y == 1) {
        __shared__ int s_kj[BLOCKFACTOR][BLOCKFACTOR];

        s_core[thr_y][thr_x] = dst[(k + thr_y) * graph_len + (k + thr_x)]; // the 64 * 64 matrix core
        s_core[thr_y][x_offset] = dst[(k + thr_y) * graph_len + (k + x_offset)];
        s_core[y_offset][thr_x] = dst[(k + y_offset) * graph_len + (k + thr_x)];
        s_core[y_offset][x_offset] = dst[(k + y_offset) * graph_len + (k + x_offset)];
        
        s_kj[thr_y][thr_x] = dst[(k + thr_y) * graph_len + (blockIdx.x * BLOCKFACTOR + thr_x)]; // left upper vertex
        s_kj[thr_y][x_offset] = dst[(k + thr_y) * graph_len + (blockIdx.x * BLOCKFACTOR + x_offset)]; // right upper
        s_kj[y_offset][thr_x] = dst[(k + y_offset) * graph_len + (blockIdx.x * BLOCKFACTOR + thr_x)]; // left bottom
        s_kj[y_offset][x_offset] = dst[(k + y_offset) * graph_len + (blockIdx.x * BLOCKFACTOR + x_offset)]; // right bottom
        __syncthreads();
        for (int p=0; p < BLOCKFACTOR; ++p) { 
            s_kj[thr_y][thr_x] = myMin(s_kj[thr_y][thr_x], s_core[thr_y][p] + s_kj[p][thr_x]);
            s_kj[thr_y][x_offset] = myMin(s_kj[thr_y][x_offset], s_core[thr_y][p] + s_kj[p][x_offset]);
            s_kj[y_offset][thr_x] = myMin(s_kj[y_offset][thr_x], s_core[y_offset][p] + s_kj[p][thr_x]);
            s_kj[y_offset][x_offset] = myMin(s_kj[y_offset][x_offset], s_core[y_offset][p] + s_kj[p][x_offset]); 
        }
        dst[(k + thr_y) * graph_len + (blockIdx.x * BLOCKFACTOR + thr_x)] = s_kj[thr_y][thr_x];
        dst[(k + thr_y) * graph_len + (blockIdx.x * BLOCKFACTOR + x_offset)] = s_kj[thr_y][x_offset];
        dst[(k + y_offset) * graph_len + (blockIdx.x * BLOCKFACTOR + thr_x)] = s_kj[y_offset][thr_x];
        dst[(k + y_offset) * graph_len + (blockIdx.x * BLOCKFACTOR + x_offset)] = s_kj[y_offset][x_offset];
    }
}
__global__ void phase3_all(const int Round, int* dst, const int graph_len) {
    if(blockIdx.x == Round || blockIdx.y == Round) {
        return;
    }
    __shared__ int s_ik[BLOCKFACTOR][BLOCKFACTOR];
    __shared__ int s_kj[BLOCKFACTOR][BLOCKFACTOR]; // padding has small effect
    __shared__ int s_ij[BLOCKFACTOR][BLOCKFACTOR]; 
    const int k = Round * BLOCKFACTOR; // start index of the real matrix
    int thr_x = threadIdx.x;
    int thr_y = threadIdx.y;
    int x_offset = thr_x + WARPSIZE;
    int y_offset = thr_y + WARPSIZE;

    s_ij[thr_y][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_ij[thr_y][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];
    s_ij[y_offset][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_ij[y_offset][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];

    s_ik[thr_y][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + thr_x)];
    s_ik[thr_y][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (k + x_offset)];
    s_ik[y_offset][thr_x] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + thr_x)];
    s_ik[y_offset][x_offset] = dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (k + x_offset)];

    s_kj[thr_y][thr_x] = dst[(k + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_kj[thr_y][x_offset] = dst[(k + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];
    s_kj[y_offset][thr_x] = dst[(k + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_kj[y_offset][x_offset] = dst[(k + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];
    __syncthreads();
    
    // #pragma unroll 64 // no use ?
    for (int p=0; p < BLOCKFACTOR; ++p) { 
        s_ij[thr_y][thr_x] = myMin(s_ij[thr_y][thr_x], s_ik[thr_y][p] + s_kj[p][thr_x]); 
        s_ij[thr_y][x_offset] = myMin(s_ij[thr_y][x_offset], s_ik[thr_y][p] + s_kj[p][x_offset]); 
        s_ij[y_offset][thr_x] = myMin(s_ij[y_offset][thr_x], s_ik[y_offset][p] + s_kj[p][thr_x]);
        s_ij[y_offset][x_offset] = myMin(s_ij[y_offset][x_offset], s_ik[y_offset][p] + s_kj[p][x_offset]); 
    }
    dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)] = s_ij[thr_y][thr_x];
    dst[(blockIdx.x * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)] = s_ij[thr_y][x_offset];
    dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)] = s_ij[y_offset][thr_x];
    dst[(blockIdx.x * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)] = s_ij[y_offset][x_offset];
}

void block_FW() {
    int round = Round_n / BLOCKFACTOR;
    
    int *d_dst;

    cudaMalloc((void **) &d_dst, sizeof(int) * Round_n * Round_n); // originally n * n
    cudaMemcpy(d_dst, dst, sizeof(int) * Round_n * Round_n, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(WARPSIZE, WARPSIZE);
    dim3 phase2_v_dim(round, 2);
    dim3 all_round(round ,round);
    
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        test_phase1<<<1, threadsPerBlock>>>(r, d_dst, Round_n);
        /* Phase 2*/     
        phase2_combine<<<phase2_v_dim, threadsPerBlock>>>(r, d_dst, Round_n);
        /* Phase 3*/ 
        phase3_all<<<all_round, threadsPerBlock>>>(r, d_dst, Round_n);
    }
    cudaMemcpy(dst, d_dst, sizeof(int) * Round_n * Round_n, cudaMemcpyDeviceToHost);
    cudaFree(d_dst);
}
void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file); // # of vertice
    fread(&m, sizeof(int), 1, file); // # of edges
    Round_n = n % BLOCKFACTOR == 0 ? n : n + (BLOCKFACTOR - n%BLOCKFACTOR);
    dst = new int[Round_n * Round_n];

    for (int i = 0; i < Round_n; ++i) {
        for (int j = 0; j < Round_n; ++j) {
            if (i == j) {
                dst[i*Round_n + j] = 0;
            } else {
                dst[i*Round_n + j] = INF;
            }
        }
    }
    int pair[3];    
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        dst[pair[0]*Round_n + pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char* outFileName) {

    FILE* outfile = fopen(outFileName, "w");
    for(int i = 0; i < n; i++) {
        fwrite(&dst[i*Round_n], sizeof(int), n, outfile); /* can save substantial time in comparison to write 1 at a time */
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {
    
    input(argv[1]); // get n and m and allocate dst memory  
    block_FW();
    output(argv[2]);

    delete []dst;
    return 0;
}