#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>

int main(int argc, char *argv[]) {
    int M = 2;
    int K = 3;
    int N = 4;

    float *h_A = (float*)malloc(sizeof(float) * M * K);
    float *h_B = (float*)malloc(sizeof(float) * K * N);
    float *h_C = (float*)malloc(sizeof(float) * M * N);
    float *h_D = (float*)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) h_A[i] = i * 1.0;
    for (int i = 0; i < K * N; i++) h_B[i] = i * 1.0;
    for (int i = 0; i < M * N; i++) h_C[i] = i * 1.0;
    for (int i = 0; i < M * N; i++) h_D[i] = 0.0;
    printf("A\t=\t"); for (int i = 0; i < M * K; i++) printf("%.1f\t", h_A[i]); printf("\n");
    printf("B\t=\t"); for (int i = 0; i < K * N; i++) printf("%.1f\t", h_B[i]); printf("\n");
    printf("C\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", h_C[i]); printf("\n");
    
    float *A, *B, *C, *D;
    cudaMalloc(&A, sizeof(float) * M * K);
    cudaMalloc(&B, sizeof(float) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);
    cudaMalloc(&D, sizeof(float) * M * N);
    cudaMemcpy(A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, h_C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t computeDesc;
    cublasLtMatmulDescCreate(&computeDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;  // D = AB + C
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, M);

    float alpha = 1.0, beta = 1.0;
    // A, B, C, D are stored as column-major  // Not use workspace
    cublasLtMatmul(
        handle, computeDesc, &alpha, A, Adesc, B, Bdesc, &beta, 
        C, Cdesc, D, Ddesc, NULL, NULL, 0, NULL
    );
    
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(computeDesc);
    cublasLtDestroy(handle);

    cudaMemcpy(h_D, D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    printf("D\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", h_D[i]); printf("\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    return 0;
}