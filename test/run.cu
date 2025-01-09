#include <cuda.h>
#include <cublasLt.h>
#include "../utils/helper.cu"
#include "../gemm/gemm.cu"

int main(int argc, char *argv[]) {
    // 若数据规模太大，而显存空间不足分配，则会导致 Segmentation fault (core dumped)
    const float alpha = 1.f, beta = 0.f;
    const int Batch = 4;
    const int M = 5120 + 111;
    const int N = 4096 + 99;
    const int K = 4096 + 77;
    const int aS = M * K, bS = K * N, cS = M * N;

    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    // h_C0 and h_C1 are used to judge same
    float *h_C0 = alloc_host_memory<float>(Batch * M * N);
    float *h_C1 = alloc_host_memory<float>(Batch * M * N);
    float *d_C_ROW = alloc_cuda_memory<float>(Batch * M * N);
    float *d_C_COL = alloc_cuda_memory<float>(Batch * M * N);
    float *d_C1 = alloc_cuda_memory<float>(Batch * M * N);
    float *d_C2 = alloc_cuda_memory<float>(Batch * M * N);
    float *d_C3 = alloc_cuda_memory<float>(Batch * M * N);
    float *d_C4 = alloc_cuda_memory<float>(Batch * M * N);

    cublasLt_sgemm(d_A, d_B, d_C_ROW, alpha, beta, M, N, K, Batch, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW);
    cublasLt_sgemm(d_A, d_B, d_C_COL, alpha, beta, M, N, K, Batch, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL);
    sgemm(d_A, d_B, d_C1, alpha, M, N, K, aS, bS, cS, GEMM_Order::RRR, Batch);
    sgemm(d_A, d_B, d_C2, alpha, M, N, K, aS, bS, cS, GEMM_Order::RRC, Batch);
    for (int b = 0; b < Batch; b++) {
        sgemm_rrr_v2(d_A + b * aS, d_B + b * bS, d_C3 + b * cS, alpha, M, N, K);
        ampere_sgemm_rrr(d_A + b * aS, d_B + b * bS, d_C4 + b * cS, alpha, M, N, K);
    }

    // 判断结果是否相等，较为耗时，若不相等，可适当调大允许误差
    #define TEST_ERROR 1.e-4
    #ifdef TEST_ERROR
    cudaMemcpy(h_C0, d_C_ROW, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C1, d_C1, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_same<float>(h_C0, h_C1, Batch * M * N, TEST_ERROR);
    
    cudaMemcpy(h_C0, d_C_COL, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C1, d_C2, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_same<float>(h_C0, h_C1, Batch * M * N, TEST_ERROR);
   
    cudaMemcpy(h_C0, d_C_ROW, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C1, d_C3, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_same<float>(h_C0, h_C1, Batch * M * N, TEST_ERROR);
    
    cudaMemcpy(h_C0, d_C_ROW, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C1, d_C4, Batch * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_same<float>(h_C0, h_C1, Batch * M * N, TEST_ERROR);
    #endif

    cudaDeviceSynchronize();
    free_memory(12, h_A, h_B, d_A, d_B, h_C0, h_C1, d_C_ROW, d_C_COL, d_C1, d_C2, d_C3, d_C4);
    return 0;
}