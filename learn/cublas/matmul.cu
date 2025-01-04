#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>
#include "../../utils/helper.cu"

int main(int argc, char *argv[]) {
    int32_t Batch = 4;
    int64_t M = 456, N = 987, K = 543;
    uint64_t workspace_bytes = 16 * 1024 * 1024;
    float alpha = 3.14, beta = 2.71;
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *h_D = alloc_host_memory<float>(Batch * M * N);
    float *h_bias = alloc_host_memory<float>(Batch * M);
    float *ret_D = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    float *d_D = alloc_cuda_memory<float>(Batch * M * N, h_D);
    float *d_bias = alloc_cuda_memory<float>(Batch * M, h_bias);
    int64_t bitmask_ld = (M + 127) / 128 * 128;
    char *bitmask = alloc_cuda_memory<char>(Batch * N * bitmask_ld / (8 * sizeof(char)), nullptr);
    void *workspace = alloc_cuda_memory<char>(workspace_bytes / sizeof(char), nullptr);

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    // 设置矩阵布局
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    cublasLtMatrixLayout_t A_layout, B_layout, C_layout, D_layout;
    cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_32F, M, K, order == CUBLASLT_ORDER_COL ? M : K);
    cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_32F, K, N, order == CUBLASLT_ORDER_COL ? K : N);
    cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_32F, M, N, order == CUBLASLT_ORDER_COL ? M : N);
    cublasLtMatrixLayoutCreate(&D_layout, CUDA_R_32F, M, N, order == CUBLASLT_ORDER_COL ? M : N);
    // 设置行主序存储或列主序存储
    cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t));
    // 设置批量数目及跨步间距
    cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(int32_t));
    int64_t A_stride = M * K, B_stride = K * N, C_stride = M * N, D_stride = M * N;
    cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &A_stride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &B_stride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &C_stride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &D_stride, sizeof(int64_t));

    // 设置矩阵乘法计算配置
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    // 设置矩阵乘法后置操作
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
    int64_t bitmask_stride = bitmask_ld * N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(cublasLtEpilogue_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(const void*));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &M, sizeof(int64_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &bitmask, sizeof(void*));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &bitmask_ld, sizeof(int64_t));
    cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &bitmask_stride, sizeof(int64_t)
    );

    // 搜索最佳的实现算法
    const int requestAlgoCount = 4;
    int returnAlgoCount = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(uint64_t)
    );
    cublasLtMatmulHeuristicResult_t heuristicResult[requestAlgoCount] = { 0 };
    cublasLtMatmulAlgoGetHeuristic(
        lt, matmul_desc, A_layout, B_layout, C_layout, D_layout, preference,
        requestAlgoCount, heuristicResult, &returnAlgoCount
    );

    // 矩阵乘法
    cublasLtMatmul(
        lt, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_C, C_layout, d_D, D_layout,
        &heuristicResult[0].algo, workspace, workspace_bytes, nullptr
    );
    cudaMemcpy(ret_D, d_D, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);

    // 主机乘法验证
    host_matmul_relu<float, col_major, col_major, col_major, col_major>(
        h_A, h_B, h_C, h_D, h_bias, alpha, beta, M, N, K, Batch
    );
    check_same<float>(h_D, ret_D, Batch * M * N, 1e-4);

    // 释放资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(D_layout);
    cublasLtMatrixLayoutDestroy(C_layout);
    cublasLtMatrixLayoutDestroy(B_layout);
    cublasLtMatrixLayoutDestroy(A_layout);
    cublasLtDestroy(lt);
    free_memory(13, h_A, h_B, h_C, h_D, h_bias, ret_D, d_A, d_B, d_C, d_D, d_bias, bitmask, workspace);
    return 0;
}