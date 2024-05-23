#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>
#include "../helper.cu"

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
    cublasStatus_t stat = cublasLtCreate(&lt);

    // 设置矩阵布局
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayout_t A_layout, B_layout, C_layout, D_layout;
    stat = cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_32F, M, K, order == CUBLASLT_ORDER_COL ? M : K);
    stat = cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_32F, K, N, order == CUBLASLT_ORDER_COL ? K : N);
    stat = cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_32F, M, N, order == CUBLASLT_ORDER_COL ? M : N);
    stat = cublasLtMatrixLayoutCreate(&D_layout, CUDA_R_32F, M, N, order == CUBLASLT_ORDER_COL ? M : N);
    // 设置行主序存储或列主序存储
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    stat = cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    // 设置批量数目及跨步间距
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    stat = cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    int64_t A_stride = M * K, B_stride = K * N, C_stride = M * N, D_stride = M * N;
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &A_stride, sizeof(A_stride));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &B_stride, sizeof(B_stride));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &C_stride, sizeof(C_stride));
    stat = cublasLtMatrixLayoutSetAttribute(D_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &D_stride, sizeof(D_stride));

    // 设置矩阵乘法计算配置
    cublasLtMatmulDesc_t matmul_desc;
    stat = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    // 设置矩阵乘法后置操作
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
    int64_t bitmask_stride = bitmask_ld * N;
    stat = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    stat = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
    stat = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &M, sizeof(M));
    stat = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &bitmask, sizeof(bitmask));
    stat = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &bitmask_ld, sizeof(bitmask_ld));
    stat = cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &bitmask_stride, sizeof(bitmask_stride)
    );

    // 搜索最佳的实现算法
    const int requestAlgoCount = 4;
    int returnAlgoCount = 0;
    cublasLtMatmulPreference_t preference;
    stat = cublasLtMatmulPreferenceCreate(&preference);
    stat = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)
    );
    cublasLtMatmulHeuristicResult_t heuristicResult[requestAlgoCount] = { 0 };
    stat = cublasLtMatmulAlgoGetHeuristic(
        lt, matmul_desc, A_layout, B_layout, C_layout, D_layout, preference,
        requestAlgoCount, heuristicResult, &returnAlgoCount
    );

    // 矩阵乘法
    stat = cublasLtMatmul(
        lt, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_C, C_layout, d_D, D_layout,
        &heuristicResult[0].algo, workspace, workspace_bytes, nullptr
    );
    cudaMemcpy(ret_D, d_D, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);

    // 主机乘法验证

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