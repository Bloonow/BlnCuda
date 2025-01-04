#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "../../utils/helper.cu"

int main(int argc, char *argv[]) {
    size_t Batch = 4, M = 456, N = 987, K = 543;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *ret_C = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        &alpha, d_A, M, M * K, d_B, K, K * N, &beta, d_C, M, M * N, Batch
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);
    cublasDestroy_v2(handle);

    host_gemm<float, col_major, col_major, col_major>(h_A, h_B, h_C, alpha, beta, M, N, K, Batch);
    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
    return 0;
}