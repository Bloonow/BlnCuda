#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "helper.cuh"

int main(int argc, char *argv[]) {
    int M = 2048, N = 512;
    float alpha = 1, beta = 1;
    float *h_A = Memory_Host<float>(M * N);
    float *h_x = Memory_Host<float>(N);
    float *h_y = Memory_Host<float>(M);
    float *d_A = Memory_Cuda<float>(M * N, h_A);
    float *d_x = Memory_Cuda<float>(N, h_x);
    float *d_y = Memory_Cuda<float>(M, h_y);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemv_v2(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1);
    cublasDestroy_v2(handle);

    float *h_y_ret = Memory_Host<float>(M);
    cudaMemcpy(h_y_ret, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    Gemv<float>(M, N, alpha, beta, h_A, h_x, h_y);
    bool same = Check_Same<float>(h_y, h_y_ret, M, 1e-3);
    printf(same ? "SAME\n" : "NOT SAME\n");

    Free_Cuda(3, d_A, d_x, d_y);
    Free_Host(4, h_A, h_x, h_y, h_y_ret);
    return 0;
}