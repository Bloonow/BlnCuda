#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "helper.cu"

int main(int argc, char *argv[]) {
    const size_t Batch = 4, M = 456, N = 987;
    const float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * N);
    float *h_x = alloc_host_memory<float>(Batch * N);
    float *h_y = alloc_host_memory<float>(Batch * M);
    float *ret_y = alloc_host_memory<float>(Batch * M);
    float *d_A = alloc_cuda_memory<float>(Batch * M * N, h_A);
    float *d_x = alloc_cuda_memory<float>(Batch * N, h_x);
    float *d_y = alloc_cuda_memory<float>(Batch * M, h_y);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    // cublasSgemv_v2(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);  // Batch = 1
    cublasSgemvStridedBatched(
        handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, M * N, d_x, 1, N, &beta, d_y, 1, M, Batch
    );
    cublasGetVector(Batch * M, sizeof(float), d_y, 1, ret_y, 1);
    cublasDestroy_v2(handle);

    host_gemv<float>(M, N, COL_MAJOR, h_A, h_x, h_y, alpha, beta, Batch);
    bool same = check_same<float>(h_y, ret_y, Batch * M, 1e-2);
    printf(same ? "SAME\n" : "NOT SAME\n");

    free_memory(7, h_A, h_x, h_y, ret_y, d_A, d_x, d_y);
    return 0;
}