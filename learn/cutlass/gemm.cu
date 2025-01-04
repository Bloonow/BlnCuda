#include <stdio.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_array.h>
#include "../../utils/helper.cu"

void gemm_demo() {
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
    cutlass::Status stat = gemm_op(
        {{M, N, K}, {d_A, M}, {d_B, K}, {d_C, M}, {d_C, M}, {alpha, beta}}
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    host_gemm<float>(M, N, K, COL_MAJOR, COL_MAJOR, COL_MAJOR, h_A, h_B, h_C, alpha, beta, 1);
    check_same<float>(h_C, ret_C,  M * N, 1.e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}

void gemm_batched_demo() {
    int Batch = 4, M = 456, N = 987, K = 543;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *ret_C = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    GemmBatched gemm_batched_op;
    cutlass::Status status = gemm_batched_op(
        {{M, N, K}, {d_A, M}, M * K, {d_B, K}, K * N, {d_C, M}, M * N, {d_C, M}, M * N, {alpha, beta}, Batch}
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);

    host_gemm<float>(M, N, K, COL_MAJOR, COL_MAJOR, COL_MAJOR, h_A, h_B, h_C, alpha, beta, Batch);
    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}

void gemm_array_demo() {
    int Batch = 4, M = 456, N = 987, K = 543;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *ret_C = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    float **dd_A_array;  cudaMalloc(&dd_A_array, Batch * sizeof(float*));
    float **dd_B_array;  cudaMalloc(&dd_B_array, Batch * sizeof(float*));
    float **dd_C_array;  cudaMalloc(&dd_C_array, Batch * sizeof(float*));
    float *d_A_array[Batch];  for(int i = 0; i < Batch; i++) { d_A_array[i] = d_A + i * M * K; }
    float *d_B_array[Batch];  for(int i = 0; i < Batch; i++) { d_B_array[i] = d_B + i * K * N; }
    float *d_C_array[Batch];  for(int i = 0; i < Batch; i++) { d_C_array[i] = d_C + i * M * N; }
    cudaMemcpy(dd_A_array, d_A_array, Batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_B_array, d_B_array, Batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_C_array, d_C_array, Batch * sizeof(float*), cudaMemcpyHostToDevice);

    using GemmArray = cutlass::gemm::device::GemmArray<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    GemmArray gemm_array_op;
    gemm_array_op(
        {{M, N, K}, dd_A_array, M, dd_B_array, K, dd_C_array, M, dd_C_array, M, {alpha, beta}, Batch}
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);

    host_gemm<float>(M, N, K, COL_MAJOR, COL_MAJOR, COL_MAJOR, h_A, h_B, h_C, alpha, beta, Batch);
    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);

    free_memory(10, h_A, h_B, h_C, ret_C, d_A, d_B, d_C, dd_A_array, dd_B_array, dd_C_array);
}

int main(int argc, char *argv[]) {
    gemm_demo();
    gemm_batched_demo();
    gemm_array_demo();
    return 0;
}