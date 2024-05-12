#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "helper.cu"

int main(int argc, char *argv[]) {
    const size_t Batch = 4, M = 1024, N = 512;
    float *h_A = allocHostMemory<float>(Batch * M * N);
    float *h_x = allocHostMemory<float>(Batch * N);
    float *h_y = allocHostMemory<float>(Batch * M);

    return 0;
}