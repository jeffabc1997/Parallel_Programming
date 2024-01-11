#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
using namespace std;
int myMerge(int left_workload, int right_workload, float left_partition[], float right_partition[], float merge_partition[], bool leftPart) { 
    // left part means that the mergePartition belongs to left process
    int swap_count = 0;
    if(leftPart) {
        int i = 0, l = 0, r = 0;
        while(i < left_workload) {
            if(r >= right_workload) { 
                merge_partition[i] = left_partition[l++];
            } else if(left_partition[l] < right_partition[r]) {
                merge_partition[i] = left_partition[l++];
            } else {
                merge_partition[i] = right_partition[r++];
                swap_count = 1;
            }
            i++;
        }
    } else { // insert data from the biggest index in right partition
        int i = right_workload - 1, l = left_workload - 1, r = right_workload - 1; 
        while(i >= 0) {
            if(l < 0) { 
                merge_partition[i] = right_partition[r--];
            } else if(left_partition[l] < right_partition[r]) {
                merge_partition[i] = right_partition[r--];
            } else {
                merge_partition[i] = left_partition[l--];
                swap_count = 1;
            }
            i--;
        }
    }
    return swap_count;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    int array_size = atoi(argv[1]); // array size
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    int root_rank = 0;

    int* startIndex = new int[numtasks];
    int* endIndex = new int[numtasks]; // if we have element 0, 1, then endIndex will be 2
    int base_workload = array_size / numtasks; 
    startIndex[0] = 0; // initialize
    endIndex[0] = (array_size % numtasks == 0) ? base_workload : base_workload + 1; // for calculate each process's workload +1 or not
    // calculate the start and end index of the original array for each partition
    for(int i = 1, tmp; i < numtasks; i++) {
        startIndex[i] = endIndex[i-1];
        tmp = (base_workload * numtasks + i >= array_size) ? base_workload : base_workload + 1; 
        endIndex[i] = startIndex[i] + tmp;
    }

    int individual_load = endIndex[rank] - startIndex[rank];
    float* partition = new float[individual_load];
    // printf("rank %d got in: %d\n", rank, individual_load);
    MPI_File input_file, output_file;
    // double start_open = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * startIndex[rank], partition, individual_load, MPI_FLOAT, MPI_STATUS_IGNORE); // (MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status * status)
    MPI_File_close(&input_file);
    // double end_open = MPI_Wtime();

    boost::sort::spreadsort::spreadsort(partition, partition + individual_load);
    
    float* another_partition;
    float* merge_partition = new float[individual_load];
    if(rank == 0) {
        another_partition = new float[endIndex[rank+1] - startIndex[rank+1]];
    } // if numtasks = 15, when rank = 14, then 14 + 1 = 15 index out of range, already solved
    else {
        another_partition = new float[endIndex[rank-1] - startIndex[rank-1]];
    }
    int i = 0, another_load;
    int swap_count = 0;
    float biggest, smallest;
    /* Well Formatted Code and modify logic of how to check whether 2 arrays should merge */
    for(int i = 0; i <= numtasks; i++) {
        swap_count = 0;
        if(i % 2 == 0) {
            // Even Phase
            if(rank % 2 == 0) { // even-indexed process
                if(rank + 1 < numtasks) { // they have a pair to merge
                    another_load = endIndex[rank+1]-startIndex[rank+1]; 
                    biggest = partition[individual_load-1];
                    MPI_Sendrecv(&biggest, 1, MPI_FLOAT, rank+1, 0, &smallest, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    if(biggest > smallest) {    // 2 local arrays have something to swap               
                    
                        MPI_Sendrecv(partition, individual_load, MPI_FLOAT, rank+1, 0, another_partition, another_load, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                        swap_count = myMerge(individual_load, another_load, partition, another_partition, merge_partition, true);
                        swap(merge_partition, partition); // swap back to partition
                    }
                    
                } else if(rank == numtasks - 1) { // this even process doesn't have pair    
                }
            } else { // odd-indexed process
                another_load = endIndex[rank-1]-startIndex[rank-1]; 
                smallest = partition[0];
                MPI_Sendrecv(&smallest, 1, MPI_FLOAT, rank-1, 0, &biggest, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                if(biggest > smallest) { // 2 local arrays have something to swap       
                    
                    MPI_Sendrecv(partition, individual_load, MPI_FLOAT, rank-1, 0, another_partition, another_load, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    swap_count = myMerge(another_load, individual_load, another_partition, partition, merge_partition, false);
                    swap(merge_partition, partition); // swap back to partition
                }
                
            }
        } else {
            // Odd Phase
            if(rank % 2 == 1) { // odd-indexed process
                if(rank + 1 < numtasks) { // they have a pair to merge
                    another_load = endIndex[rank+1]-startIndex[rank+1];
                    biggest = partition[individual_load-1];
                    MPI_Sendrecv(&biggest, 1, MPI_FLOAT, rank+1, 0, &smallest, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    if(biggest > smallest) {  // 2 local arrays have something to swap          
                        
                        MPI_Sendrecv(partition, individual_load, MPI_FLOAT, rank+1, 0, another_partition, another_load, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        swap_count += myMerge(individual_load, another_load, partition, another_partition, merge_partition, true);
                        swap(merge_partition, partition); // swap back to partition
                    }
                    
                } else if(rank == numtasks - 1) { // the process doesn't have a even-indexed process to merge with

                }
            } else { // even-indexed process
                if(rank != 0) { // 0-indexed process can do nothing
                    another_load = endIndex[rank-1]-startIndex[rank-1];
                    smallest = partition[0];
                    MPI_Sendrecv(&smallest, 1, MPI_FLOAT, rank-1, 0, &biggest, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    if(biggest > smallest) { // 2 local arrays have something to swap             
                        
                        MPI_Sendrecv(partition, individual_load, MPI_FLOAT, rank-1, 0, another_partition, another_load, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        swap_count += myMerge(another_load, individual_load, another_partition, partition, merge_partition, false);
                        swap(merge_partition, partition); // swap back to partition
                    }
                } else { // rank = 0   
                }        
            }
        }
    }   

    
    
    // double start_close = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * startIndex[rank], partition, individual_load, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    // double end_close = MPI_Wtime();
    // if(rank < 3) {
    //     std::cout << "RANK: " << rank <<"The process took " << end_close - start_close << " seconds to write files." << std::endl;
    // }
    MPI_Finalize();
    return 0;
}

