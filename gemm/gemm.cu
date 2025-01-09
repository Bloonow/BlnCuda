#pragma once

#include <cuda.h>
#include "sgemm_128x128.cu"
#include "sgemm_128x128_v2.cu"
#include "sgemm_32x32.cu"
#include "sgemm_32x32_slicek.cu"
#include "sgemm_32x32_splitk.cu"
#include "ampere_sgemm.cu"
#include "hgemm_128x128.cu"

void sgemm(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS,
    const GEMM_Order order, const uint32_t batchCount
) {
    // 对于 128x128 的 kernel 来说，限制每个 SM 启动 2 个，而 4090 有 76 个 SM，支持 152 个 kernel 同时运行，此处取 64 作为界限
    if (M >= 128 && N >= 128 && ((M + 127) / 128) * ((N + 127) / 128) * batchCount >= 64) {
        sgemm_128x128_8x8::sgemm_cuda(A, B, C, alpha, M, N, K, aS, bS, cS, order, batchCount);
    } else {
        if (K <= 48) {
            sgemm_32x32_4x4::sgemm_cuda(A, B, C, alpha, M, N, K, aS, bS, cS, order, batchCount);
        } else if (K >= 256 && (M < 1024 && N < 1024)) {
            // 当 K 比较大时使用 SplitK + ReduceK 算法
            sgemm_32x32_4x8_SplitK::sgemm_cuda(A, B, C, alpha, M, N, K, aS, bS, cS, order, batchCount);
        } else {
            sgemm_32x32_4x8_SliceK::sgemm_cuda(A, B, C, alpha, M, N, K, aS, bS, cS, order, batchCount);
        }
    }
}

void sgemm_rrr_v2(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 127) / 128, (M + 127) / 128, 1);
    sgemm_128x128_8x8::sgemm_rrr_128x128x8_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K);
}

void ampere_sgemm_rrr(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 255) / 256, (M + 127) / 128, 1);
    sgemm_128x256_16x8::ampere_sgemm_rrr_128x256x8_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K);
}

#ifndef __CUBLASLT_WARPPER__
#define __CUBLASLT_WARPPER__
#include <cublasLt.h>
void cublasLt_sgemm(
    const float *A, const float *B, float *C, const float alpha, const float beta,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t batchCount,
    const cublasLtOrder_t orderA, const cublasLtOrder_t orderB, const cublasLtOrder_t orderC,
    const size_t workspaceSize = 4 * 1024 * 1024
) {
    cublasLtHandle_t blasLt = NULL;
    cublasLtMatmulDesc_t compute = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    void *workspace = NULL;  // For supporting SplitK Algorithm when batchCount is one.
    cudaMalloc(&workspace, workspaceSize);
    cublasLtCreate(&blasLt);
    cublasLtMatmulDescCreate(&compute, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, orderA == CUBLASLT_ORDER_ROW ? K : M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, orderB == CUBLASLT_ORDER_ROW ? N : K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, orderC == CUBLASLT_ORDER_ROW ? N : M);
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderC, sizeof(orderC));
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    int64_t aS = M * K, bS = K * N, cS = M * N;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &aS, sizeof(aS));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &bS, sizeof(bS));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &cS, sizeof(cS));
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    cublasLtMatmulAlgoGetHeuristic(blasLt, compute, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    cublasLtMatmul(
        blasLt, compute, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, C, Cdesc,
        &heuristicResult.algo, workspace, workspaceSize, NULL
    );
    if (workspace)  cudaFree(workspace);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Cdesc)      cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc)      cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc)      cublasLtMatrixLayoutDestroy(Adesc);
    if (compute)    cublasLtMatmulDescDestroy(compute);
    if (blasLt)     cublasLtDestroy(blasLt);
}
#endif // __BL_CUBLASLT_WARPPER__