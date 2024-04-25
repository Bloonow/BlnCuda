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
    
    double **A = (double**)malloc(sizeof(double*) * batchCount);
    double **B = (double**)malloc(sizeof(double*) * batchCount);
    double **C = (double**)malloc(sizeof(double*) * batchCount);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        A[bidx] = (double*)malloc(sizeof(double) * M * K);
        B[bidx] = (double*)malloc(sizeof(double) * K * N);
        C[bidx] = (double*)malloc(sizeof(double) * M * N);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        for (int i = 0; i < M * K; i++) A[bidx][i] = i;    // 行主序(M,K)矩阵，仅逻辑上转置，可看成列主序(K,M)矩阵
        for (int i = 0; i < K * N; i++) B[bidx][i] = i;    // 行主序(K,N)矩阵，仅逻辑上转置，可看成列主序(N,K)矩阵
        for (int i = 0; i < M * N; i++) C[bidx][i] = 0;    // 行主序(M,N)矩阵，仅逻辑上转置，可看成列主序(N,M)矩阵
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("A[%d]\t=\t", bidx); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[bidx][i]); printf("\n");
        printf("B[%d]\t=\t", bidx); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[bidx][i]); printf("\n");
    }

    double **d_A_ptr = (double**)malloc(sizeof(double*) * batchCount);
    double **d_B_ptr = (double**)malloc(sizeof(double*) * batchCount);
    double **d_C_ptr = (double**)malloc(sizeof(double*) * batchCount);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaMalloc(&d_A_ptr[bidx], sizeof(double) * M * K);
        cudaMalloc(&d_B_ptr[bidx], sizeof(double) * K * N);
        cudaMalloc(&d_C_ptr[bidx], sizeof(double) * M * N);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cublasSetMatrix(K, M, sizeof(double), A[bidx], K, d_A_ptr[bidx], K);  // 看成列主序(K,M)矩阵时，应该使用的参数
        cublasSetMatrix(N, K, sizeof(double), B[bidx], N, d_B_ptr[bidx], N);  // 看成列主序(N,K)矩阵时，应该使用的参数
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_A[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_A_ptr[bidx], M * K); cudaDeviceSynchronize();
        printf("d_B[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_B_ptr[bidx], K * N); cudaDeviceSynchronize();
    }

    double alpha = 1.0;
    double beta = 0.0;
    // 因为d_A_ptr[bidx],d_B_ptr[bidx],d_C_ptr[bidx]变量虽然指向设备地址，但其本身是位于主机内存上的，
    // 而cublas<t>gemmBatched()函数需要根据地址访问d_A_ptr[bidx]等变量，故需要再将d_A_ptr[bidx]等本身复制到设备内存上
    double **d_A;
    double **d_B;
    double **d_C;
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaMalloc((double**)&d_A, sizeof(double*) * batchCount);
        cudaMalloc((double**)&d_B, sizeof(double*) * batchCount);
        cudaMalloc((double**)&d_C, sizeof(double*) * batchCount); 
    }
    cudaMemcpy(d_A, d_A_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, d_B_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, d_C_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    // 求C，列主序
    cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M, batchCount);

    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_C[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_C_ptr[bidx], M * N); cudaDeviceSynchronize();
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
        cublasGetMatrix(M, N, sizeof(double), d_C_ptr[bidx], M, C[bidx], M);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("C[%d]\t=\t", bidx); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[bidx][i]); printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaFree(d_A_ptr[bidx]);
        cudaFree(d_B_ptr[bidx]);
        cudaFree(d_C_ptr[bidx]);
        free(A[bidx]);
        free(B[bidx]);
        free(C[bidx]);
    }
    free(d_A_ptr);
    free(d_B_ptr);
    free(d_C_ptr);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}