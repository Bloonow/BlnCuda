#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>
#include "../learn/helper.cu"

int main(int argc, char *argv[]) {
    int32_t Batch = 4;
    int64_t M = 64, N = 64, K = 16384;
    uint64_t workspace_bytes = 16 * 1024 * 1024;
    float alpha = 1.0, beta = 0;
    float *h_A = alloc_host_memory<float>(Batch * M * K);
    float *h_B = alloc_host_memory<float>(Batch * K * N);
    float *h_C = alloc_host_memory<float>(Batch * M * N);
    float *ret_C = alloc_host_memory<float>(Batch * M * N);
    float *d_A = alloc_cuda_memory<float>(Batch * M * K, h_A);
    float *d_B = alloc_cuda_memory<float>(Batch * K * N, h_B);
    float *d_C = alloc_cuda_memory<float>(Batch * M * N, h_C);
    void *workspace = alloc_cuda_memory<char>(workspace_bytes / sizeof(char), nullptr);

    cublasLtHandle_t lt;
    cublasStatus_t stat = cublasLtCreate(&lt);

    // 设置矩阵布局
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayout_t A_layout, B_layout, C_layout;
    stat = cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_32F, M, K, order == CUBLASLT_ORDER_COL ? M : K);
    stat = cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_32F, K, N, order == CUBLASLT_ORDER_COL ? K : N);
    stat = cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_32F, M, N, order == CUBLASLT_ORDER_COL ? M : N);
    // 设置行主序存储或列主序存储
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    // 设置批量数目及跨步间距
    int64_t A_stride = M * K, B_stride = K * N, C_stride = M * N;
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &A_stride, sizeof(A_stride));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &B_stride, sizeof(B_stride));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &C_stride, sizeof(C_stride));
    stat = cublasLtMatrixLayoutSetAttribute(A_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    stat = cublasLtMatrixLayoutSetAttribute(B_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));
    stat = cublasLtMatrixLayoutSetAttribute(C_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch));

    // 设置矩阵乘法计算配置
    cublasLtMatmulDesc_t matmul_desc;
    stat = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // has used cublasLtMatrixLayoutSetAttribute() to set BatchCount and Strided
    const int requestAlgoCount = 4;
    int returnAlgoCount = 0;
    cublasLtMatmulPreference_t preference;
    stat = cublasLtMatmulPreferenceCreate(&preference);
    stat = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)
    );
    cublasLtMatmulHeuristicResult_t heuristicResult[requestAlgoCount] = { 0 };
    stat = cublasLtMatmulAlgoGetHeuristic(
        lt, matmul_desc, A_layout, B_layout, C_layout, C_layout, preference,
        requestAlgoCount, heuristicResult, &returnAlgoCount
    );
    cublasLtMatmulAlgo_t algo = heuristicResult[0].algo;
    int32_t splitK = 16;  // set the split number for using SplitK algorithm
    stat = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK));
    stat = cublasLtMatmul(
        lt, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_C, C_layout, d_C, C_layout,
        &algo, workspace, workspace_bytes, nullptr
    );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("[Error][%d], %s\n", stat, cublasLtGetStatusString(stat));
    }
    
    cudaMemcpy(ret_C, d_C, sizeof(float) * Batch * M * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(C_layout);
    cublasLtMatrixLayoutDestroy(B_layout);
    cublasLtMatrixLayoutDestroy(A_layout);
    cublasLtDestroy(lt);
    free_memory(8, h_A, h_B, h_C, ret_C, d_A, d_B, d_C, workspace);
    return 0;
}