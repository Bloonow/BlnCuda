#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint32_t cvt_smem_addr(const void *smem_ptr) {
    // 共享内存指针转换为整数类型的地址
    uint32_t smem_addr;
    asm volatile (
        "{\n"
        ".reg .u64 u64addr;\n"
        "cvta.to.shared.u64 u64addr, %1;\n"
        "cvt.u32.u64 %0, u64addr;\n"
        "}\n"
        : "=r"(smem_addr)
        : "l"(smem_ptr)
    );
    return smem_addr;
}

template <int round>
__launch_bounds__(16, 1)  /* 一个SM执行最多1个线程块，一个线程块最多16个线程 */
__global__ void smem_latency_kernel(const uint32_t* src, uint32_t* dst, uint32_t* clk) {
    __shared__ uint32_t smem_buf[16];
    smem_buf[threadIdx.x] = src[threadIdx.x];
    uint32_t smem_addr = cvt_smem_addr(smem_buf + threadIdx.x);

    uint32_t start, stop;
    // bar.sync是线程同步指令，0指示无条件同步
    // %%clock用于获取clock寄存器，表示GPU运行时的时钟计数，单位通常是GPU的时钟周期
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < round; i++) {
        // 所读取的下一个地址，依赖于当期读取的值，这保证下一条ldg指令必须在当前ldg指令完成后才能执行，使之无法被掩盖
        asm volatile (
            "ld.shared.b32 %0, [%0];\n"
            : "+r"(smem_addr)
            :
            : "memory"
        );
    }
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory"
    );
    clk[threadIdx.x] = stop - start;

    if (smem_addr == 0) {
        *dst = smem_addr;  // 防止编译器优化
    }
}

int main(int argc, char* argv[]) {
    const int round = 50;  // 执行多少次lds指令

    uint32_t *h_src;
    cudaMallocHost(&h_src, 16 * sizeof(uint32_t));
    for (int i = 0; i < 16; i++) h_src[i] = i * sizeof(uint32_t);
    uint32_t *src, *dst;
    uint32_t *clk;
    cudaMalloc(&src, 16 * sizeof(uint32_t));
    cudaMalloc(&dst, sizeof(uint32_t));
    cudaMemcpy(src, h_src, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMalloc(&clk, 16 * sizeof(uint32_t));

    const int warmup_amount = 100;
    // 预热，占用L0指令缓存和L1指令缓存
    for (int i = 0; i < warmup_amount; i++) {
        smem_latency_kernel<round><<<1, 16>>>(src, dst, clk);
    }
    // 测试共享内存延迟
    smem_latency_kernel<round><<<1, 16>>>(src, dst, clk);

    uint32_t h_clk[16];
    cudaMemcpy(h_clk, clk, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("Shared memory latency: %u cycle\n", h_clk[0] / round);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(clk);
    cudaFreeHost(h_src);
    return 0;
}