#include <stdio.h>
#include <cuda.h>
#include <cublasLt.h>
#include "../learn/helper.cu"

int main(int argc, char *argv[]) {
    size_t M = 64, N = 64;
    int8_t alpha = 1, beta = 0;
    int8_t *h1 = alloc_host_memory_increment<int8_t>(M * N);
    int8_t *h2 = alloc_host_memory<int8_t>(M * N);
    int8_t *d1 = alloc_cuda_memory<int8_t>(M * N, h1);
    int8_t *d2 = alloc_cuda_memory<int8_t>(M * N, nullptr);

    cublasStatus_t stat;
    cublasLtHandle_t lt;
    stat = cublasLtCreate(&lt);
    cublasLtMatrixTransformDesc_t transformDesc;
    stat = cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_8I);

    cublasLtMatrixLayout_t d1_layout, d2_layout;
    stat = cublasLtMatrixLayoutCreate(&d1_layout, CUDA_R_8I, M, N, M);
    stat = cublasLtMatrixLayoutCreate(&d2_layout, CUDA_R_8I, M, N, 32 * 8 * 8);
    cublasLtOrder_t target_order = CUBLASLT_ORDER_COL4_4R2_8C;
    stat = cublasLtMatrixLayoutSetAttribute(d2_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &target_order, sizeof(target_order));
    stat = cublasLtMatrixTransform(lt, transformDesc, &alpha, d1, d1_layout, &beta, nullptr, nullptr, d2, d2_layout, nullptr);
    cudaError_t err = cudaMemcpy(h2, d2, M * N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    err = cudaDeviceSynchronize();
    printf("P1!\n");
    err = cudaDeviceSynchronize();
    printf("P2!\n");
    
    cublasLtMatrixLayoutDestroy(d1_layout);
    cublasLtMatrixLayoutDestroy(d2_layout);
    cublasLtMatrixTransformDescDestroy(transformDesc);
    cublasLtDestroy(lt);
    free_memory(4, h1, h2, d1, d2);
    return 0;
}
