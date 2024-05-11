#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "helper.cuh"

int main(int argc, char *argv[]) {
    int M = 2048, N = 512;
    float alpha = 1, beta = 1;
    float *h_A = memory_host<float>(M * N);
    float *h_x = memory_host<float>(N);
    float *h_y = memory_host<float>(M);
    float *d_A = memory_cuda<float>(M * N, h_A);
    float *d_x = memory_cuda<float>(N, h_x);
    float *d_y = memory_cuda<float>(M, h_y);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemv_v2(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    cublasDestroy_v2(handle);

    float *h_y_ret = memory_host<float>(M);
    cudaMemcpy(h_y_ret, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    gemv<float>(M, N, alpha, beta, h_A, h_x, h_y);
    bool same = check_same<float>(h_y, h_y_ret, M, 1e-3);
    printf(same ? "SAME\n" : "NOT SAME\n");

    free_cuda(3, d_A, d_x, d_y);
    free_host(4, h_A, h_x, h_y, h_y_ret);
    return 0;
}