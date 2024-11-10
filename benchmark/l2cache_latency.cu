#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint32_t ldg_cg(const void* ldg_ptr) {
    uint32_t ret;
    // ld.global指令从设备全局内存中进行读取
    // cg表示仅使用L2缓存，而绕过L1缓存
    // b32表示将数据的类型解析为32位的Bits(untyped)
    // memory表示该内联汇编会影响内存状态，编译器应该避免优化掉相关内存操作，这在循环或并行执行时尤为重要
    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(ret)
        : "l"(ldg_ptr)
        : "memory"
    );
    return ret;
}

template <int round>
__launch_bounds__(32, 1)  /* 一个SM执行最多1个线程块，一个线程块最多32个线程 */
__global__ void l2_latency_kernel(const uint32_t* src, uint32_t* dst, uint32_t* clk) {
    const char* ldg_ptr = reinterpret_cast<const char*>(src + threadIdx.x);
    uint32_t val;
    val = ldg_cg(ldg_ptr);
    ldg_ptr += val;

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
        // IADD/IMAD/XMAD等指令的延迟远小于L2缓存的延迟，可以忽略不记
        val = ldg_cg(ldg_ptr);
        ldg_ptr += val;
    }
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory"
    );
    clk[threadIdx.x] = stop - start;

    if (val == 0) {
        *dst = val;  // 防止编译器优化
    }
}

int main(int argc, char* argv[]) {
    // 仅使用一个Warp中的32个线程即可
    const int stride = 128;  // 一个线程的两条相邻ldg指令所读取内存地址之间的跨步
    const int round = 10;    // 执行多少次ldg指令
    const int data_bytes = stride * (round + 1);
    static_assert(stride >= 32 * sizeof(uint32_t) && stride % sizeof(uint32_t) == 0, "'stride' is invalid");

    uint32_t *src, *dst;
    uint32_t *clk;
    cudaMalloc(&src, data_bytes);
    cudaMalloc(&dst, sizeof(uint32_t));
    cudaMalloc(&clk, 32 * sizeof(uint32_t));
    uint32_t* h_src;
    cudaMallocHost(&h_src, data_bytes);
    for (int i = 0; i < data_bytes / sizeof(uint32_t); i++) h_src[i] = stride;
    cudaMemcpy(src, h_src, data_bytes, cudaMemcpyHostToDevice);

    const int warmup_amount = 100;
    // 预热，占用L0指令缓存和L1指令缓存，以及L2缓存
    for (int i = 0; i < warmup_amount; i++) {
        l2_latency_kernel<round><<<1, 32>>>(src, dst, clk);
    }
    // 测试L2缓存的延迟
    l2_latency_kernel<round><<<1, 32>>>(src, dst, clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("L2 cache latency: %u cycles\n", h_clk[0] / round);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(clk);
    cudaFreeHost(h_src);
    return 0;
}