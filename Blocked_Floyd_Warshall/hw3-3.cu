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
    if(blockIdx.y == 0) { // vertical
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
    } else if (blockIdx.y == 1) { // horizon
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
__global__ void phase3_all(const int Round, int* dst, const int graph_len, const int blockx_offset) {
    // if(blockIdx.x == Round || blockIdx.y == Round) {
    //     return;
    // }
    __shared__ int s_ik[BLOCKFACTOR][BLOCKFACTOR];
    __shared__ int s_kj[BLOCKFACTOR][BLOCKFACTOR]; // padding has small effect
    __shared__ int s_ij[BLOCKFACTOR][BLOCKFACTOR]; 
    const int k = Round * BLOCKFACTOR; // start index of the real matrix

    
    int thr_x = threadIdx.x;
    int thr_y = threadIdx.y;
    int x_offset = thr_x + WARPSIZE;
    int y_offset = thr_y + WARPSIZE;
    int blockx = blockIdx.x + blockx_offset;
    //     if(blockx_offset == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("Round: %i, x, y (%i, %i)\n", Round, (blockx * BLOCKFACTOR + thr_y), (blockIdx.y * BLOCKFACTOR + thr_x));
    // }
    s_ij[thr_y][thr_x] = dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_ij[thr_y][x_offset] = dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];
    s_ij[y_offset][thr_x] = dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)];
    s_ij[y_offset][x_offset] = dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)];

    s_ik[thr_y][thr_x] = dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (k + thr_x)];
    s_ik[thr_y][x_offset] = dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (k + x_offset)];
    s_ik[y_offset][thr_x] = dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (k + thr_x)];
    s_ik[y_offset][x_offset] = dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (k + x_offset)];

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
    dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)] = s_ij[thr_y][thr_x];
    dst[(blockx * BLOCKFACTOR + thr_y) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)] = s_ij[thr_y][x_offset];
    dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + thr_x)] = s_ij[y_offset][thr_x];
    dst[(blockx * BLOCKFACTOR + y_offset) * graph_len + (blockIdx.y * BLOCKFACTOR + x_offset)] = s_ij[y_offset][x_offset];
}

void block_FW() {
    int round = Round_n / BLOCKFACTOR;
    int *d_dst[2];


    dim3 threadsPerBlock(WARPSIZE, WARPSIZE);
    
    const int num_gpus = 2;
    int block_width = round;
    int block_cut = (round % 2 == 0) ? (round / 2) : (round / 2) + 1;
    #pragma omp parallel num_threads(num_gpus)
    {
    int tid = omp_get_thread_num();
    int peer_device = (tid+1) % 2;

    int block_start = block_cut * tid; // which row the device starts working
    int block_end = (tid == 0) ? block_cut : block_width;
    int work_blocks = block_end - block_start; // how many blocks (in row) the device handles
    
    // printf("BLock cut: %i round: %i Rn: %i work_blocks %i\n", block_cut, round, Round_n, work_blocks);
    cudaSetDevice(tid);
    cudaMalloc((void **) &d_dst[tid], sizeof(int) * Round_n * Round_n); 
    cudaMemcpy(d_dst[tid], dst, sizeof(int) * Round_n * Round_n, cudaMemcpyHostToDevice);
    
    /* if we have the row 1 updated in phase 3 in round 0, then we can calculate round 1's pivot row (row 1) */
    /* we only need to focus on the pivot cols in our own partition. that's enough for phase 3 */
    dim3 phase2_v_dim(round, 2);
    dim3 partition(work_blocks ,round);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        test_phase1<<<1, threadsPerBlock>>>(r, d_dst[tid], Round_n);
        /* Phase 2*/     
        phase2_combine<<<phase2_v_dim, threadsPerBlock>>>(r, d_dst[tid], Round_n);
        
        /* Phase 3*/ 
        phase3_all<<<partition, threadsPerBlock>>>(r, d_dst[tid], Round_n, block_start);

        #pragma omp barrier
        if(tid == 0) {
            if(r+1 < block_cut) {
                // send pivot, row r+1
                // printf("id: %i, round: %i row_block sent: %i\n", tid, r, r+1);
                cudaMemcpy(&d_dst[peer_device][(r+1)* BLOCKFACTOR *Round_n], &d_dst[tid][(r+1)* BLOCKFACTOR *Round_n], sizeof(int) * Round_n*BLOCKFACTOR, cudaMemcpyDeviceToDevice);
            } 
        } else { // tid = 1
            if((r+1 >= block_cut) && (r + 1 < round)){
                // printf("id: %i, round: %i row_block sent: %i\n", tid, r, r+1);
                cudaMemcpy(&d_dst[peer_device][(r+1)* BLOCKFACTOR * Round_n], &d_dst[tid][(r+1)* BLOCKFACTOR *Round_n], sizeof(int) * Round_n*BLOCKFACTOR, cudaMemcpyDeviceToDevice);
            } 
        }
        if(r+1 == round) {
            cudaMemcpy(&dst[block_start * BLOCKFACTOR * Round_n], &d_dst[tid][block_start* BLOCKFACTOR *Round_n], sizeof(int) * (work_blocks * BLOCKFACTOR) * Round_n, cudaMemcpyDeviceToHost);
        }
        #pragma omp barrier
    }
    
    cudaFree(d_dst[tid]);
    }    
}
void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file); // # of vertice
    fread(&m, sizeof(int), 1, file); // # of edges
    Round_n = n % BLOCKFACTOR == 0 ? n : n + (BLOCKFACTOR - n%BLOCKFACTOR); // n % BLOCKFACTOR = 0
    // printf("test: %i\n", (BLOCKFACTOR - n%BLOCKFACTOR));
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