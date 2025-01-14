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

static constexpr uint32_t M = 1024 + 55;
static constexpr uint32_t N = 1024 + 33;
static constexpr uint32_t K = 512 + 11;
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
    Gemm compute_op;
    // 所需要的参数，执行时动态构造
    cutlass::gemm::GemmCoord problem_size = { M, N, K };
    cutlass::TensorRef<const ElementA, LayoutA> ref_A = { d_A, K };
    cutlass::TensorRef<const ElementB, LayoutB> ref_B = { d_B, K };
    cutlass::TensorRef<const ElementC, LayoutC> ref_C = { d_C, N };
    cutlass::TensorRef<ElementAccumulator, LayoutC> ref_D = { d_D, N };
    EpilogueOutputOp::Params epilogue = { alpha, beta };
    Gemm::Arguments args = {
        problem_size, ref_A, ref_B, ref_C, ref_D, epilogue
    };
    cutlass::Status status = compute_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("%s\n", cutlass::cutlassGetStatusString(status));
        exit(-1);
    }
    // 所需的工作空间
    size_t workspace_bytes = compute_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
    // 调用方式之一
    // status = compute_op.initialize(args, workspace.get());
    // status = compute_op.run(nullptr);
    // 调用方式之二，实际上是operator()内部先调用initialize()方法再调用run()方法
    compute_op(args, workspace.get(), nullptr);
    // 释放所分配的工作空间
    workspace.reset();

    host_gemm<float, row_major, float, col_major, float, row_major>(h_A, h_B, h_C, h_D, alpha, beta, M, N, K, 1);
    cudaMemcpy(ret_D, d_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    check_same<float>(h_D, ret_D,  M * N, 1.e-4);
    free_memory(9, h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D, ret_D);
}

void demo_gemm_batched() {
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *h_D = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    float *d_D = alloc_cuda_memory<float>(Batch * M * N, h_D);
    float *ret_D = alloc_host_memory<float>(Batch * M * N);

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
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    constexpr int Stages = 2;
    constexpr int AlignmentA = 1;
    constexpr int AlignmentB = 1;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, AlignmentA, AlignmentB, Operator
    >;
    // 矩阵乘法类的实例对象
    GemmBatched compute_op;
    // 所需要的参数，执行时动态构造
    cutlass::gemm::GemmCoord problem_size = { M, N, K };
    cutlass::TensorRef<const ElementA, LayoutA> ref_A = { d_A, K };
    int64_t stride_A = M * K;
    cutlass::TensorRef<const ElementB, LayoutB> ref_B = { d_B, K };
    int64_t stride_B = K * N;
    cutlass::TensorRef<const ElementC, LayoutC> ref_C = { d_C, N };
    int64_t stride_C = M * N;
    cutlass::TensorRef<ElementAccumulator, LayoutC> ref_D = { d_D, N };
    int64_t stride_D = M * N;
    EpilogueOutputOp::Params epilogue = { alpha, beta };
    int batch_count = Batch;
    GemmBatched::Arguments args = {
        problem_size, ref_A, stride_A, ref_B, stride_B, ref_C, stride_C, ref_D, stride_D, epilogue, batch_count
    };
    cutlass::Status status = compute_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("%s\n", cutlass::cutlassGetStatusString(status));
        exit(-1);
    }
    // 所需的工作空间
    size_t workspace_bytes = compute_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
    // 调用方式之一
    // status = compute_op.initialize(args, workspace.get());
    // status = compute_op.run(nullptr);
    // 调用方式之二，实际上是operator()内部先调用initialize()方法再调用run()方法
    compute_op(args, workspace.get(), nullptr);
    // 释放所分配的工作空间
    workspace.reset();

    host_gemm<float, row_major, float, col_major, float, row_major>(h_A, h_B, h_C, h_D, alpha, beta, M, N, K, Batch);
    cudaMemcpy(ret_D, d_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    check_same<float>(h_D, ret_D,  M * N, 1.e-4);
    free_memory(9, h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D, ret_D);
}

void demo_gemm_array() {
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *h_D = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    float *d_D = alloc_cuda_memory<float>(Batch * M * N, h_D);
    float *ret_D = alloc_host_memory<float>(Batch * M * N);
    // 数组元素是设备内存上的每个矩阵数据的地址
    float **h_A_devAddrs = (float**)malloc(Batch * sizeof(float*));
    float **h_B_devAddrs = (float**)malloc(Batch * sizeof(float*));
    float **h_C_devAddrs = (float**)malloc(Batch * sizeof(float*));
    float **h_D_devAddrs = (float**)malloc(Batch * sizeof(float*));
    for(int i = 0; i < Batch; i++) {
        h_A_devAddrs[i] = d_A + i * M * K;
        h_B_devAddrs[i] = d_B + i * K * N;
        h_C_devAddrs[i] = d_C + i * M * N;
        h_D_devAddrs[i] = d_D + i * M * N;
    }
    float **d_A_devAddrs = alloc_cuda_memory<float*>(Batch, h_A_devAddrs);
    float **d_B_devAddrs = alloc_cuda_memory<float*>(Batch, h_B_devAddrs);
    float **d_C_devAddrs = alloc_cuda_memory<float*>(Batch, h_C_devAddrs);
    float **d_D_devAddrs = alloc_cuda_memory<float*>(Batch, h_D_devAddrs);

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
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    constexpr int Stages = 2;
    constexpr int AlignmentA = 1;
    constexpr int AlignmentB = 1;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using GemmArray = cutlass::gemm::device::GemmArray<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, AlignmentA, AlignmentB, Operator
    >;
    // 矩阵乘法类的实例对象
    GemmArray compute_op;
    // 所需要的参数，执行时动态构造
    cutlass::gemm::GemmCoord problem_size = { M, N, K };
    EpilogueOutputOp::Params epilogue = { alpha, beta };
    GemmArray::Arguments args = {
        problem_size, d_A_devAddrs, K, d_B_devAddrs, K, d_C_devAddrs, N, d_D_devAddrs, N, epilogue, Batch
    };
    cutlass::Status status = compute_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("%s\n", cutlass::cutlassGetStatusString(status));
        exit(-1);
    }
    // 所需的工作空间
    size_t workspace_bytes = compute_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
    // 调用方式之一
    // status = compute_op.initialize(args, workspace.get());
    // status = compute_op.run(nullptr);
    // 调用方式之二，实际上是operator()内部先调用initialize()方法再调用run()方法
    compute_op(args, workspace.get(), nullptr);
    // 释放所分配的工作空间
    workspace.reset();

    host_gemm<float, row_major, float, col_major, float, row_major>(h_A, h_B, h_C, h_D, alpha, beta, M, N, K, Batch);
    cudaMemcpy(ret_D, d_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    check_same<float>(h_D, ret_D,  M * N, 1.e-4);
    free_memory(
        17, h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D, ret_D, 
        h_A_devAddrs, h_B_devAddrs, h_C_devAddrs, h_D_devAddrs,
        d_A_devAddrs, d_B_devAddrs, d_C_devAddrs, d_D_devAddrs
    );
}

void demo_gemm_tensor_core() {
    const uint32_t M = (::M + 127) / 128 * 128;
    const uint32_t N = (::N + 127) / 128 * 128;
    const uint32_t K = (::K + 127) / 128 * 128;
    cutlass::half_t *h_A = alloc_host_memory<cutlass::half_t>(M * K);
    cutlass::half_t *h_B = alloc_host_memory<cutlass::half_t>(K * N);
    float *h_C = alloc_host_memory<float>(M * N);
    float *h_D = alloc_host_memory<float>(M * N);
    cutlass::half_t *d_A = alloc_cuda_memory<cutlass::half_t>(M * K, h_A);
    cutlass::half_t *d_B = alloc_cuda_memory<cutlass::half_t>(K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(M * N, h_C);
    float *d_D = alloc_cuda_memory<float>(M * N, h_D);
    float *ret_D = alloc_host_memory<float>(M * N);

    // 编译时静态构造的矩阵乘法类
    using ElementA = cutlass::half_t; using LayoutA = cutlass::layout::RowMajor;
    using ElementB = cutlass::half_t; using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = float;           using LayoutC = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm75;
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    // 对于TensorCore实现而言，通常使用128/sizeof_bits<Type>::value作为对齐值，因为通常会使用uint4进行向量化访存
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator
    >;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
    constexpr int Stages = 2;
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr bool SplitKSerial = false;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using GemmTensorCore = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, AlignmentA, AlignmentB, SplitKSerial, Operator
    >;
    // 矩阵乘法类的实例对象
    GemmTensorCore compute_op;
    // 所需要的参数，执行时动态构造
    cutlass::gemm::GemmCoord problem_size = { M, N, K };
    cutlass::TensorRef<const ElementA, LayoutA> ref_A = { d_A, K };
    cutlass::TensorRef<const ElementB, LayoutB> ref_B = { d_B, K };
    cutlass::TensorRef<const ElementC, LayoutC> ref_C = { d_C, N };
    cutlass::TensorRef<ElementAccumulator, LayoutC> ref_D = { d_D, N };
    EpilogueOutputOp::Params epilogue = { alpha, beta };
    GemmTensorCore::Arguments args = {
        problem_size, ref_A, ref_B, ref_C, ref_D, epilogue
    };
    cutlass::Status status = compute_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("%s\n", cutlass::cutlassGetStatusString(status));
        exit(-1);
    }
    // 所需的工作空间
    size_t workspace_bytes = compute_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);
    // 调用方式之一
    // status = compute_op.initialize(args, workspace.get());
    // status = compute_op.run(nullptr);
    // 调用方式之二，实际上是operator()内部先调用initialize()方法再调用run()方法
    compute_op(args, workspace.get(), nullptr);
    // 释放所分配的工作空间
    workspace.reset();

    host_gemm<cutlass::half_t, row_major, cutlass::half_t, col_major, float, row_major>(
        h_A, h_B, h_C, h_D, alpha, beta, M, N, K, 1
    );
    cudaMemcpy(ret_D, d_D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    check_same<float>(h_D, ret_D,  M * N, 1.e-3);
    free_memory(9, h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D, ret_D);
}

int main(int argc, char *argv[]) {
    demo_gemm();
    demo_gemm_batched();
    demo_gemm_array();
    demo_gemm_tensor_core();
    return 0;
}