#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_array.h>
#include "../../utils/helper.cu"

static constexpr uint32_t M = 5120;
static constexpr uint32_t N = 5120;
static constexpr uint32_t K = 2560;
static constexpr uint32_t Batch = 4;
static constexpr float alpha = 3.14;
static constexpr float beta = 2.71;

void gemm_demo() {
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

    host_gemm<float, col_major, col_major, col_major>(h_A, h_B, h_C, alpha, beta, M, N, K, 1);
    check_same<float>(h_C, ret_C,  M * N, 1.e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}

void gemm_batched_demo() {
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

    host_gemm<float, col_major, col_major, col_major>(h_A, h_B, h_C, alpha, beta, M, N, K, Batch);
    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}

void gemm_array_demo() {
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

    host_gemm<float, col_major, col_major, col_major>(h_A, h_B, h_C, alpha, beta, M, N, K, Batch);
    check_same<float>(h_C, ret_C, Batch * M * N, 1e-4);

    free_memory(10, h_A, h_B, h_C, ret_C, d_A, d_B, d_C, dd_A_array, dd_B_array, dd_C_array);
}

void gemm_tensor_core_demo() {
    const uint32_t M = (::M + 127) / 128 * 128;
    const uint32_t N = (::N + 127) / 128 * 128;
    const uint32_t K = (::K + 127) / 128 * 128;
    cutlass::half_t *h_A = alloc_host_memory<cutlass::half_t>(M * K);
    cutlass::half_t *h_B = alloc_host_memory<cutlass::half_t>(K * N);
    float *h_C = alloc_host_memory<float>(M * N);
    float *ret_C = alloc_host_memory<float>(M * N);
    cutlass::half_t *d_A = alloc_cuda_memory<cutlass::half_t>(M * K, h_A);
    cutlass::half_t *d_B = alloc_cuda_memory<cutlass::half_t>(K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(M * N, h_C);

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using GemmTensorCore = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape
    >;
    GemmTensorCore gemm_tensor_core_op;
    cutlass::Status stat = gemm_tensor_core_op(
        {{M, N, K}, {d_A, K}, {d_B, K}, {d_C, N}, {d_C, N}, {alpha, beta}}
    );
    cudaMemcpy(ret_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    host_gemm<row_major, col_major, row_major>(h_A, h_B, h_C, alpha, beta, M, N, K, 1);
    check_same<float>(h_C, ret_C, M * N, 1.e-4);

    free_memory(7, h_A, h_B, h_C, ret_C, d_A, d_B, d_C);
}


int main(int argc, char *argv[]) {
    gemm_demo();
    gemm_batched_demo();
    gemm_array_demo();
    gemm_tensor_core_demo();
    return 0;
}