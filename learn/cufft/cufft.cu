#include <stdio.h>
#include <cuda.h>
#include <cufft.h>

__global__ void scale_kernel(cufftComplex* data, float factor, const int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid].x *= factor;
        data[tid].y *= factor;
    }
}

int main(int argc, char *argv[]) {
    int batch = 8;
    int N1 = 256, N2 = 128;
    cufftReal *h_input = (cufftReal*)malloc(batch * N1 * N2 * sizeof(cufftReal));
    cufftReal *h_reslut = (cufftReal*)malloc(batch * N1 * N2 * sizeof(cufftReal));
    for (int i = 0; i < batch * N1 * N2; i++) h_input[i] = i * 1.f / batch;
    cufftReal *input, *result;
    cufftComplex *intermediate;
    cudaMalloc((cufftReal**)(&input), batch * N1 * N2 * sizeof(cufftReal));
    cudaMalloc((cufftReal**)(&result), batch * N1 * N2 * sizeof(cufftReal));
    cudaMalloc((cufftComplex**)(&intermediate), batch * N1 * (N2 / 2 + 1) * sizeof(cufftComplex));
    cudaMemcpy(input, h_input, batch * N1 * N2 * sizeof(cufftReal), cudaMemcpyHostToDevice);

    int N[2] = {N1, N2};
    cufftHandle plan2D_r2c, plan2D_c2r;
    cufftCreate(&plan2D_r2c);
    cufftCreate(&plan2D_c2r);
    // 构建plan配置
    cufftPlanMany(&plan2D_r2c, 2, N, nullptr, 1, N1 * N2, nullptr, 1, N1 * (N2 / 2 + 1), CUFFT_R2C, batch);
    cufftPlanMany(&plan2D_c2r, 2, N, nullptr, 1, N1 * (N2 / 2 + 1), nullptr, 1, N1 * N2, CUFFT_C2R, batch);
    cufftExecR2C(plan2D_r2c, input, intermediate);   // 正变换
    // 因为傅里叶逆变换需要除以N，故在变换之前先进行标准化，也可以在变换之后进行标准化
    scale_kernel<<<(batch * N1 * (N2 / 2 + 1) + 127) / 128, 128>>>(
        intermediate, 1.f / (N1 * N2), batch * N1 * (N2 / 2 + 1)
    );
    cufftExecC2R(plan2D_c2r, intermediate, result);  // 逆变换
    cufftDestroy(plan2D_r2c);
    cufftDestroy(plan2D_c2r);

    bool all_same = true;
    cudaMemcpy(h_reslut, result, batch * N1 * N2 * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch * N1 * N2; i++) {
        if (abs(h_input[i] - h_reslut[i]) > 1.e-3) {
            all_same = false;
            break;
        }
    }
    printf("The data are %s.\n", all_same ? "all same" : "not all same");

    cudaFree(input);
    cudaFree(result);
    cudaFree(intermediate);
    free(h_input);
    free(h_reslut);
    return 0;
}