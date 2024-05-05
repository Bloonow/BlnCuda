#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>

int main(int argc, char *argv[]) {
    int M = 2;
    int K = 3;
    int N = 2;
    int32_t batch = 2;
    float *h_A = (float*)malloc(sizeof(float) * M * K * batch);  // X, input
    float *h_B = (float*)malloc(sizeof(float) * K * N);          // W, weight
    float *h_C = (float*)malloc(sizeof(float) * 1 * N);          // B, bias
    float *h_D = (float*)malloc(sizeof(float) * M * N * batch);  // Y = X W + B
    for (int i = 0; i < M * K * batch; i++) h_A[i] = (i % (M * K)) * 1.0;
    for (int i = 0; i < K * N; i++)         h_B[i] = i * 1.0;
    for (int i = 0; i < 1 * N; i++)         h_C[i] = i * 0.1;
    for (int i = 0; i < M * N * batch; i++) h_D[i] = 0.0;
    printf("A\t=\t");
    for (int i = 0; i < M * K * batch; i++) printf("%.1f\t", h_A[i]);
    printf("\n");
    printf("B\t=\t");
    for (int i = 0; i < K * N; i++) printf("%.1f\t", h_B[i]);
    printf("\n");
    printf("C\t=\t");
    for (int i = 0; i < 1 * N; i++) printf("%.1f\t", h_C[i]);
    printf("\n");
    
    float *A, *B, *C, *D;
    cudaMalloc(&A, sizeof(float) * M * K * batch);
    cudaMalloc(&B, sizeof(float) * K * N);
    cudaMalloc(&C, sizeof(float) * 1 * N);
    cudaMalloc(&D, sizeof(float) * M * N * batch);
    cudaMemcpy(A, h_A, sizeof(float) * M * K * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, h_C, sizeof(float) * 1 * N, cudaMemcpyHostToDevice);

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t computeDesc;
    cublasLtMatmulDescCreate(&computeDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;  // row-major
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, 0);  // broadcast
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, N);
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(
        Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(
        Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    int64_t Astride = M * K;
    int64_t Bstride = 0;
    int64_t Cstride = 0;
    int64_t Dstride = M * N;
    cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Astride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Bstride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Cstride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(
        Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Dstride, sizeof(int64_t));

    float alpha = 1.0, beta = 1.0;
    // Not use workspace
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

    cudaMemcpy(h_D, D, sizeof(float) * M * N * batch, cudaMemcpyDeviceToHost);
    printf("D\t=\t"); 
    for (int i = 0; i < M * N * batch; i++) printf("%.1f\t", h_D[i]); 
    printf("\n");
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