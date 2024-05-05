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
    
    double *A = (double*)malloc(sizeof(double) * M * K);
    double *B = (double*)malloc(sizeof(double) * K * N);
    double *C = (double*)malloc(sizeof(double) * M * N);
    for (int i = 0; i < M * K; i++) A[i] = i;  // 行主序(M,K)矩阵，可看成列主序(K,M)矩阵
    for (int i = 0; i < K * N; i++) B[i] = i;  // 行主序(K,N)矩阵，可看成列主序(N,K)矩阵
    for (int i = 0; i < M * N; i++) C[i] = 0;  // 行主序(M,N)矩阵，可看成列主序(N,M)矩阵
    printf("A\t=\t"); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[i]); printf("\n");
    printf("B\t=\t"); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[i]); printf("\n");

    double alpha = 1.0;
    double beta = 0.0;
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * M * K);
    cudaMalloc(&d_B, sizeof(double) * K * N);
    cudaMalloc(&d_C, sizeof(double) * M * N);
    // cublasSetMatrix(M, K, sizeof(double), A, M, d_A, M);  // 列主序(M,K)的情况
    // cublasSetMatrix(K, N, sizeof(double), B, K, d_B, K);  // 列主序(M,K)的情况
    cublasSetMatrix(K, M, sizeof(double), A, K, d_A, K);  // 看成列主序(K,M)矩阵
    cublasSetMatrix(N, K, sizeof(double), B, N, d_B, N);  // 看成列主序(N,K)矩阵
    printf("d_A\t=\t"); dis_matrix<<<1,1>>>(d_A, M * K); cudaDeviceSynchronize();
    printf("d_B\t=\t"); dis_matrix<<<1,1>>>(d_B, K * N); cudaDeviceSynchronize();

    // 列主序情况，m,n,k参数分别表示(M,K)的op(A)矩阵，(K,N)的op(B)矩阵，
    // lda,ldb为A,B的前导维数M,K，ldc为C的前导维数
    // cublasDgemm(
    //     handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
    //     &alpha, d_A, M, d_B, K, &beta, d_C, M
    // );

    // 行主序情况，要将A,B看成列主序，(K,M)矩阵与(N,K)矩阵无法相乘，
    // 故需要转置op(A)和op(B)，转置后分别为(M,K)矩阵和(K,N)矩阵，
    // lda,ldb为A,B的前导维数，因将A,B看成列主序的(K,M)矩阵和(N,K)矩阵，故分别为K,N，
    // ldc为C的前导维数，因为无法控制矩阵C进行转置，
    // 故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
    cublasDgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
        &alpha, d_A, K, d_B, N, &beta, d_C, M
    );
    
    printf("d_C\t=\t"); dis_matrix<<<1,1>>>(d_C, M * N); cudaDeviceSynchronize();
    // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
    cublasGetMatrix(M, N, sizeof(double), d_C, M, C, M);
    printf("C\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[i]); printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}