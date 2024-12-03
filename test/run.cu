#include <cuda.h>
#include <cublasLt.h>

#include "check.cu"
#include "../gemm/sgemm.cu"

using check::matrix_init;
using check::matrix_same;

int main(int argc, char *argv[]) {
    // 若数据规模太大，而显存空间不足分配，则会导致 Segmentation fault (core dumped)
    const float alpha = 1.f, beta = 0.f;
    const int batchCount = 1;
    const int M = 5120;
    const int N = 4096;
    const int K = 2048;
    const int aS = M * K, bS = K * N, cS = M * N;

    float *A, *B, *C0_ROW, *C0_COL, *C1, *C2, *C3;
    matrix_init(&A, M, K, batchCount);
    matrix_init(&B, K, N, batchCount);
    matrix_init(&C0_ROW, M, N, batchCount, 0.f);
    matrix_init(&C0_COL, M, N, batchCount, 0.f);
    matrix_init(&C1, M, N, batchCount, 0.f);
    matrix_init(&C2, M, N, batchCount, 0.f);
    matrix_init(&C3, M, N, batchCount, 0.f);

    // 是否测试正确性
    #define TEST_ERROR 1.e-5

    cublasLt_sgemm(A, B, C0_ROW, alpha, beta, M, N, K, batchCount, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW);
    cublasLt_sgemm(A, B, C0_COL, alpha, beta, M, N, K, batchCount, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL);
    sgemm(A, B, C1, alpha, M, N, K, aS, bS, cS, GEMM_Order::RRR, batchCount);
    sgemm(A, B, C2, alpha, M, N, K, aS, bS, cS, GEMM_Order::RRC, batchCount);
    sgemm_rrr_v2(A, B, C3, alpha, M, N, K);
    // 判断结果是否相等，较为耗时，若不相等，可适当调大允许误差
    #ifdef TEST_ERROR
    matrix_same(C0_ROW, C1, M, N, batchCount, TEST_ERROR);
    matrix_same(C0_COL, C2, M, N, batchCount, TEST_ERROR);
    matrix_same(C0_ROW, C3, M, N, batchCount, TEST_ERROR);
    #endif

    cudaDeviceSynchronize();
    cudaFree(A);
    cudaFree(B);
    cudaFree(C0_ROW);
    cudaFree(C0_COL);
    cudaFree(C1);
    cudaFree(C2);
    cudaFree(C3);
    return 0;
}