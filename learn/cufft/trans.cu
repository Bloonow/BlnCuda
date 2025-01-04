#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include "../../utils/helper.cu"

__global__ void scale_kernel(cufftComplex* data, float factor, const int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid].x *= factor;
        data[tid].y *= factor;
    }
}

int main(int argc, char *argv[]) {
    int Batch = 8;
    int N1 = 256, N2 = 128;
    cufftReal *h_input = (cufftReal*)malloc(Batch * N1 * N2 * sizeof(cufftReal));
    cufftReal *h_reslut = (cufftReal*)malloc(Batch * N1 * N2 * sizeof(cufftReal));
    for (int i = 0; i < Batch * N1 * N2; i++) h_input[i] = i * 1.f / Batch;
    cufftReal *d_input, *d_result;
    cufftComplex *d_temp;
    cudaMalloc((cufftReal**)(&d_input), Batch * N1 * N2 * sizeof(cufftReal));
    cudaMalloc((cufftReal**)(&d_result), Batch * N1 * N2 * sizeof(cufftReal));
    cudaMalloc((cufftComplex**)(&d_temp), Batch * N1 * (N2 / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(d_input, h_input, Batch * N1 * N2 * sizeof(cufftReal), cudaMemcpyHostToDevice);

    cufftHandle plan2D_r2c, plan2D_c2r;
    cufftCreate(&plan2D_r2c);
    cufftCreate(&plan2D_c2r);
    // 构建plan配置
    int Ranks[2] = { N1, N2 };
    int R_nembed[2] = { N1, N2 };
    int C_nembed[2] = { N1, N2 / 2 + 1 };
    cufftPlanMany(&plan2D_r2c, 2, Ranks, R_nembed, 1, N1 * N2, C_nembed, 1, N1 * (N2 / 2 + 1), CUFFT_R2C, Batch);
    cufftPlanMany(&plan2D_c2r, 2, Ranks, C_nembed, 1, N1 * (N2 / 2 + 1), R_nembed, 1, N1 * N2, CUFFT_C2R, Batch);
    // 正变换
    cufftExecR2C(plan2D_r2c, d_input, d_temp);
    // 标准化，在逆变换之前进行
    scale_kernel<<<(Batch * N1 * (N2 / 2 + 1) + 127) / 128, 128>>>(d_temp, 1.f / (N1 * N2), Batch * N1 * (N2 / 2 + 1));
    // 逆变换
    cufftExecC2R(plan2D_c2r, d_temp, d_result);
    cufftDestroy(plan2D_r2c);
    cufftDestroy(plan2D_c2r);

    cudaMemcpy(h_reslut, d_result, Batch * N1 * N2 * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    check_same<cufftReal>(h_reslut, h_input, Batch * N1 * N2, 1.e-3);

    free_memory(5, h_input, h_reslut, d_input, d_result, d_temp);
    return 0;
}