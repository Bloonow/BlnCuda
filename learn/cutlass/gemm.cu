#include <stdio.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include "../helper.cu"

int main(int argc, char *argv[]) {
    int M = 456, N = 987, K = 543;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(M * K);
    float *h_B = alloc_host_memory<float>(K * N);
    float *h_C = alloc_host_memory<float>(M * N);
    float *ret_C = alloc_host_memory<float>(M * N);
    float *d_A = alloc_cuda_memory<float>(M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(M * N, h_C);

    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70
    >;
    Gemm gemm_op;
    cutlass::Status stat = gemm_op({{M, N, K}, {d_A, M}, {d_B, K}, {d_C, M}, {d_C, M}, {alpha, beta}});
    cudaMemcpy(ret_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    host_gemm<float>(M, N, K, COL_MAJOR, COL_MAJOR, COL_MAJOR, h_A, h_B, h_C, alpha, beta, 1);
    bool same = check_same<float>(h_C, ret_C,  M * N, 1.e-4);
    printf(same ? "|| SAME ||\n" : "|| NOT SAME ||\n");

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
    return 0;
}