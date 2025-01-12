#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "../utils/helper.cu"

static constexpr uint32_t M = 512 + 33;
static constexpr uint32_t N = 512 + 22;
static constexpr uint32_t K = 256 + 11;
static constexpr uint32_t Batch = 4;
static constexpr float alpha = 3.14;
static constexpr float beta = 2.71;

void demo_cublasSgemmStridedBatched() {
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *ret_C = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        &alpha, d_A, M, M * K, d_B, K, K * N, &beta, d_C, M, M * N, Batch
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    host_gemm<float, col_major, float, col_major, float, col_major>(
        h_A, h_B, h_C, h_C, alpha, beta, M, N, K, Batch
    );

    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);
    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}

void demo_cublasSgemvStridedBatched() {
    float *h_A = alloc_host_memory<float>(Batch * M * N);
    float *h_x = alloc_host_memory<float>(Batch * N);
    float *h_y = alloc_host_memory<float>(Batch * M);
    float *ret_y = alloc_host_memory<float>(Batch * M);
    float *d_A = alloc_cuda_memory<float>(Batch * M * N, h_A);
    float *d_x = alloc_cuda_memory<float>(Batch * N, h_x);
    float *d_y = alloc_cuda_memory<float>(Batch * M, h_y);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemvStridedBatched(
        handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, M * N, d_x, 1, N, &beta, d_y, 1, M, Batch
    );
    cudaMemcpy(ret_y, d_y, sizeof(float) * Batch * M, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    host_gemv<float, col_major>(h_A, h_x, h_y, alpha, beta, M, N, Batch);

    check_same<float>(h_y, ret_y, Batch * M, 1e-4);
    free_memory(7, h_A, h_x, h_y, ret_y, d_A, d_x, d_y);
}

int main(int argc, char *argv[]) {
    demo_cublasSgemmStridedBatched();
    demo_cublasSgemvStridedBatched();
    return 0;
}