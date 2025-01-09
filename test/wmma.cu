#include "../utils/helper.cu"
#include "../gemm/hgemm_128x128.cu"
#include "../gemm/gemm.cu"

int main(int argc, char *argv[]) {
    int M = 5120 * 1, N = 5120 * 1, K = 2560 * 1;
    float alpha = 1.5, beta = 2.78;
    half* h_A = alloc_host_memory<half>(M * K);
    half* h_B = alloc_host_memory<half>(K * N);
    float* h_C = alloc_host_memory<float>(M * N);
    float *ret_D0 = alloc_host_memory<float>(M * N);
    float *ret_D1 = alloc_host_memory<float>(M * N);
    half* d_A = alloc_cuda_memory<half>(M * K, h_A);
    half* d_B = alloc_cuda_memory<half>(K * N, h_B);
    float* d_C = alloc_cuda_memory<float>(M * N, h_C);
    float* d_D = alloc_cuda_memory<float>(M * N);

    wmma_hgemm_m16n16k16::wmma_hgemm_rcr_cuda(d_A, d_B, d_C, d_D, alpha, beta, M, N, K);
    cudaMemcpy(ret_D0, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cublasLt_hgemm(d_A, d_B, d_C, alpha, beta, M, N, K, 1, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_ROW);
    cudaMemcpy(ret_D1, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // host_gemm<row_major, col_major, row_major>(h_A, h_B, h_C, 1, 0, M, N, K, 1);
    check_same<float>(ret_D0, ret_D1, M * N, 1.e-3);

    free_memory(9, h_A, h_B, h_C, ret_D0, ret_D1, d_A, d_B, d_C, d_D);
    return 0;
}