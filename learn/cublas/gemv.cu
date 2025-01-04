#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "../../utils/helper.cu"

int main(int argc, char *argv[]) {
    size_t Batch = 4, M = 456, N = 987;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * N);
    float *h_x = alloc_host_memory<float>(Batch * N);
    float *h_y = alloc_host_memory<float>(Batch * M);
    float *ret_y = alloc_host_memory<float>(Batch * M);
    float *d_A = alloc_cuda_memory<float>(Batch * M * N, h_A);
    float *d_x = alloc_cuda_memory<float>(Batch * N, h_x);
    float *d_y = alloc_cuda_memory<float>(Batch * M, h_y);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemvStridedBatched(
        handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, M * N, d_x, 1, N, &beta, d_y, 1, M, Batch
    );
    cudaMemcpy(ret_y, d_y, sizeof(float) * Batch * M, cudaMemcpyDeviceToHost);
    cublasDestroy_v2(handle);

    host_gemv<float, col_major>(h_A, h_x, h_y, alpha, beta, M, N, Batch);
    check_same<float>(h_y, ret_y, Batch * M, 1e-3);

    free_memory(7, h_A, h_x, h_y, ret_y, d_A, d_x, d_y);
    return 0;
}