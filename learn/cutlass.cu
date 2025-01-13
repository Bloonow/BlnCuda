#include <stdio.h>
#include <cuda.h>
#include "../utils/helper.cu"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/util/device_memory.h>

static constexpr uint32_t M = 512;
static constexpr uint32_t N = 512;
static constexpr uint32_t K = 256;
static constexpr uint32_t Batch = 4;
static constexpr float alpha = 3.14;
static constexpr float beta = 2.71;

void demo_gemm() {
    float *h_A = alloc_host_memory<float>(M * K);
    float *h_B = alloc_host_memory<float>(K * N);
    float *h_C = alloc_host_memory<float>(M * N);
    float *h_D = alloc_host_memory<float>(M * N);
    float *d_A = alloc_cuda_memory<float>(M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(M * N, h_C);
    float *d_D = alloc_cuda_memory<float>(M * N, h_D);
    float *ret_D = alloc_host_memory<float>(M * N);

    // 编译时静态构造的矩阵乘法类
    using ElementA = float; using LayoutA = cutlass::layout::RowMajor;
    using ElementB = float; using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = float; using LayoutC = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassSimt;
    using ArchTag = cutlass::arch::Sm70;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
    constexpr int Stages = 2;
    constexpr int AlignmentA = 1;
    constexpr int AlignmentB = 1;
    constexpr bool SplitKSerial = false;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, AlignmentA, AlignmentB, SplitKSerial, Operator
    >;
    // 矩阵乘法类的实例对象
    Gemm ComputeOp;
    // 所需要的参数，执行时动态构造
    cutlass::gemm::GemmCoord problem_size = { M, N, K };
    cutlass::TensorRef<const ElementA, LayoutA> ref_A = { d_A, K };
    cutlass::TensorRef<const ElementB, LayoutB> ref_B = { d_B, K };
    cutlass::TensorRef<const ElementC, LayoutC> ref_C = { d_C, N };
    cutlass::TensorRef<ElementAccumulator, LayoutC> ref_D = { d_D, N };
    EpilogueOutputOp::Params epilogue = { alpha, beta };
    Gemm::Arguments args(problem_size, ref_A, ref_B, ref_C, ref_D, epilogue);
    cutlass::Status status = ComputeOp.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("%s\n", cutlass::cutlassGetStatusString(status));
        exit(-1);
    }
    // 所需的工作空间
    size_t workspace_bytes = ComputeOp.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
    // 调用方式之一
    // status = ComputeOp.initialize(args, workspace.get());
    // status = ComputeOp.run(nullptr);
    // 调用方式之二，实际上是operator()内部先调用initialize()方法再调用run()方法
    ComputeOp(args, workspace.get(), nullptr);
    // 释放所分配的工作空间
    workspace.reset();

    host_gemm<float, row_major, float, col_major, float, row_major>(h_A, h_B, h_C, h_D, alpha, beta, M, N, K, 1);
    cudaMemcpy(ret_D, d_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    check_same<float>(h_D, ret_D,  M * N, 1.e-4);
    free_memory(9, h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D, ret_D);
}

/*

void demo_gemm_batched() {
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

void demo_gemm_array() {
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

void demo_gemm_tensor_core() {
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

    // 对于TensorCore实现而言，通常使用128/sizeof_bits<Type>::value作为对齐值，因为通常会使用uint4进行向量化访存

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

*/

int main(int argc, char *argv[]) {
    demo_gemm();
    // demo_gemm_batched();
    // demo_gemm_array();
    // demo_gemm_tensor_core();
    return 0;
}