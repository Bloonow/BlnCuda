#include <cstdio>
#include <cstdint>

template <int round>
__launch_bounds__(4, 1)  /* 一个SM执行最多1个线程块，一个线程块最多4个线程 */
__global__ void l1_latency_kernel(void** src, void** dst, uint32_t* clk) {
    void** ldg_ptr = src + threadIdx.x;
    // 预热，占用L1缓存
    for (int i = 0; i < round; i++) {
        // ld.global指令从设备全局内存中进行读取
        // nc表示通过非一致纹理缓存（non-coherent texture cache）访问设备全局内存空间
        // b64表示将数据的类型解析为64位的Bits(untyped)
        // +l中的+加号表示该操作数既是输入又是输出
        // memory表示该内联汇编会影响内存状态，编译器应该避免优化掉相关内存操作，这在循环或并行执行时尤为重要
        asm volatile (
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr)
            :
            : "memory"
        );
    }

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
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr)
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

    if (ldg_ptr == nullptr) {
        *dst = ldg_ptr;  // 防止编译器优化
    }
}

int main() {
    const int thread_num = 4;  // 仅使用一个Warp中的4个线程即可
    const int round = 10;      // 执行多少次ldg指令

    void** src;
    void** dst;
    uint32_t* clk;
    cudaMalloc(&src, thread_num * sizeof(void*));
    cudaMalloc(&dst, sizeof(void*));
    cudaMalloc(&clk, thread_num * sizeof(uint32_t));
    void **h_src;
    cudaMallocHost(&h_src, thread_num * sizeof(void*));
    for (int i = 0; i < thread_num; i++) h_src[i] = src + i;
    // src[i]存储的是src[i]自己的地址
    cudaMemcpy(src, h_src, thread_num * sizeof(void*), cudaMemcpyHostToDevice);

    const int warmup_amount = 100;
    // 预热，占用指令Cache
    for (int i = 0; i < warmup_amount; i++) {
        l1_latency_kernel<round><<<1, thread_num>>>(src, dst, clk);
    }
    // 测试L1缓存的延迟
    l1_latency_kernel<round><<<1, thread_num>>>(src, dst, clk);

    uint32_t h_clk[thread_num];
    cudaMemcpy(h_clk, clk, thread_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("L1 cache latency: %u cycles\n", h_clk[0] / round);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(clk);
    cudaFreeHost(h_src);
    return 0;
}