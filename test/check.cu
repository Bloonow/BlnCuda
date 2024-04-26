#pragma once

#include <stdio.h>
#include <cuda.h>

namespace check {

__global__ void matrix_set_value_kernel(
    float *mat, const int M, const int N, const int batchCount = 1, const float alpha = 1.f
) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    for (int id = 0; id < batchCount; ++id) {
        if (ty < M && tx < N) {
            mat[id * M * N + ty * N + tx] = (id + 3.14f) * sin(tx + ty / tx + 1.e-6f) + (id - 3.14f) * cos(ty - tx / ty + 1.e-6f);
            mat[id * M * N + ty * N + tx] *= alpha;
        }
    }
}

__global__ void matrix_same_kernel(
    const float *mat1, const float *mat2, const int M, const int N, const int batchCount = 1, const float err = 1.e-6f
) {
    printf("Matrix Same Checking... ");
    bool same = true;
    for (int id = 0; id < batchCount; ++id) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (abs(mat1[id * M * N + i * N + j] - mat2[id * M * N + i * N + j]) > err) {
                    printf(
                        "|| No Same [%d] | [%d][%d] || mat1 = %f, mat2 = %f\n", 
                        id, i, j, mat1[id * M * N + i * N + j], mat2[id * M * N + i * N + j]
                    );
                    return;
                }
            }
        }
    }
    if (same) {
        printf("|| Same ||\n");
    }
}

__global__ void matrix_display_kernek(
    float *mat, const int M, const int N,
    const int M_start = 0, const int N_start = 0,
    const int M_span = 8, const int N_span = 8, const int batch_id = 0
) {
    int M_end = M_start + M_span; if (M_end > M) M_end = M;
    int N_end = N_start + N_span; if (N_end > N) N_end = N;
    printf("\n=== Matrix Display @ Batch %d, [%d:%d][%d:%d]===\n", batch_id, M_start, M_end, N_start, N_end);
    for (int i = M_start; i < M_end; ++i) {
        for (int j = N_start; j < N_end; ++j) {
            printf("%.2f\t", mat[batch_id * M * N + i * N + j]);
        }
        printf("\n");
    }
    printf("=== Display Over ====\n\n");
}

void matrix_init(
    float **mat, const int M, const int N, const int batchCount = 1, const float alpha = 1.f
) {
    cudaMalloc((void**)(mat), batchCount * M * N * sizeof(float));
    const dim3 block_size(32, 32, 1);
    const dim3 grid_size((N + 31) / 32, (M + 31) / 32, 1);
    matrix_set_value_kernel<<<grid_size, block_size>>>(*mat, M, N, batchCount, alpha);
}

void matrix_same(
    const float *mat1, const float *mat2, const int M, const int N, const int batchCount = 1, const float err = 1.e-6f
) {
    matrix_same_kernel<<<1, 1>>>(mat1, mat2, M, N, batchCount, err);
}

void matrix_display(
    float *mat, const int M, const int N,
    const int M_start = 0, const int N_start = 0,
    const int M_span = 8, const int N_span = 8, const int batch_id = 0
) {
    matrix_display_kernek<<<1, 1>>>(mat, M, N, M_start, N_start, M_span, N_span, batch_id);
}

} // namespace check
