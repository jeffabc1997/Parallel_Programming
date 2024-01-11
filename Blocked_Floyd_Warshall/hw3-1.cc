
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define MAX_WEIGHT 1073741823

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>

#include <omp.h>
#include <iostream>

using namespace std;
int** adj_mat;
int num_vertex, num_edge;
void read_bin_to_adj(fstream& bin_in) {
    for(int i = 0; i < num_vertex; i++) {
        for(int j = 0; j < num_vertex; j++) {
            if(i == j) {
                adj_mat[i][j] = 0;
            } else {
                adj_mat[i][j] = MAX_WEIGHT;
            }
        }
    }
    for(int i = 0, src, dst, wt; i < num_edge; i++) {
        bin_in.read((char*)&src,sizeof(int));
        bin_in.read((char*)&dst,sizeof(int));
        bin_in.read((char*)&wt,sizeof(int));
        adj_mat[src][dst] = wt;
    }
}
void output(char* outFileName) {

    FILE* outfile = fopen(outFileName, "w");
    for(int i = 0; i < num_vertex; i++) {
        fwrite(adj_mat[i], sizeof(int), num_vertex, outfile); /* can save substantial time in comparison to write 1 at a time */
    }
    fclose(outfile);
}

int main(int argc, char** argv) {
    
    string input_file(argv[1]);
    string output_file(argv[2]);
    std::fstream bin_in(input_file, std::ios_base::binary|std::ios_base::in);
    
    /* get first 2 integers |V| and |E| */
    bin_in.read((char*)&num_vertex,sizeof(int));
    bin_in.read((char*)&num_edge,sizeof(int));
    adj_mat = new int*[num_vertex];
    for(int i = 0; i < num_vertex; i++) {
        adj_mat[i] = new int[num_vertex];
    }

    read_bin_to_adj(bin_in);
    bin_in.close();
    
    for(int k = 0, dst = MAX_WEIGHT; k < num_vertex; k++) {
        bool res = false;
        #pragma omp parallel 
        {
            #pragma omp for schedule(static) nowait collapse(2)
            for(int i=0; i < num_vertex; i++) {
                for(int j = 0; j < num_vertex; j++) {

                    adj_mat[i][j] = min(adj_mat[i][j], (adj_mat[i][k] + adj_mat[k][j]));
                }
            }
        
        }
    }

    output(argv[2]);
    for(int i = 0; i < num_vertex; i++) {
        delete []adj_mat[i];
    }
    return 0;
}