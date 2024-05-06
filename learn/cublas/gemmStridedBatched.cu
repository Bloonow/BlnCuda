#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

__global__ void dis_matrix(double *d_M, int eles) {
    for (int i = 0; i < eles; i++) printf("%.1f\t", d_M[i]); printf("\n");
}

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    int M = 2;
    int K = 3;
    int N = 4;
    int batchCount = 2;
    
    double *A = (double*)malloc(sizeof(double) * M * K * batchCount);
    double *B = (double*)malloc(sizeof(double) * K * N * batchCount);
    double *C = (double*)malloc(sizeof(double) * M * N * batchCount);
    for (int i = 0; i < M * K * batchCount; i++) A[i] = i;  // 行主序(M,K)矩阵，可看成列主序(K,M)矩阵
    for (int i = 0; i < K * N * batchCount; i++) B[i] = i;  // 行主序(K,N)矩阵，可看成列主序(N,K)矩阵
    for (int i = 0; i < M * N * batchCount; i++) C[i] = 0;  // 行主序(M,N)矩阵，可看成列主序(N,M)矩阵
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("A[%d]\t=\t", bidx); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[bidx*M*K+i]);
        printf("\n");
        printf("B[%d]\t=\t", bidx); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[bidx*K*N+i]);
        printf("\n");
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * M * K * batchCount);
    cudaMalloc(&d_B, sizeof(double) * K * N * batchCount);
    cudaMalloc(&d_C, sizeof(double) * M * N * batchCount);
    cublasSetMatrix(K, M * batchCount, sizeof(double), A, K, d_A, K);  // 看成列主序(K,M)矩阵
    cublasSetMatrix(N, K * batchCount, sizeof(double), B, N, d_B, N);  // 看成列主序(N,K)矩阵
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_A[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_A[bidx*M*K], M * K); cudaDeviceSynchronize();
        printf("d_B[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_B[bidx*K*N], K * N); cudaDeviceSynchronize();
    }

    double alpha = 1.0;
    double beta = 0.0;
    // 求C，列主序
    cublasDgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
        &alpha, d_A, K, M*K, d_B, N, K*N, &beta, d_C, M, M*N, batchCount
    );

    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_C[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_C[bidx*M*N], M * N); cudaDeviceSynchronize();
    }

    for (int bidx = 0; bidx < batchCount; bidx++) {
        // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
        cublasGetMatrix(M, N * batchCount, sizeof(double), d_C, M, C, M);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("C[%d]\t=\t", bidx); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[bidx*M*N+i]);
        printf("\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}