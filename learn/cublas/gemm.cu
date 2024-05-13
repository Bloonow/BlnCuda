#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "../helper.cu"

int main(int argc, char *argv[]) {
    const size_t Batch = 4, M = 456, N = 987, K = 543;
    const float alpha = 3.14, beta = 2.71;
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
    cublasGetMatrix(M, N, sizeof(float), d_C, M, ret_C, M);
    cublasDestroy_v2(handle);

    host_gemm<float>(M, N, K, COL_MAJOR, COL_MAJOR, COL_MAJOR, h_A, h_B, h_C, alpha, beta, Batch);
    bool same = check_same<float>(h_C, ret_C, 1e-4);
    printf(same ? "SAME\n" : "NOT SAME\n");

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
    return 0;
}