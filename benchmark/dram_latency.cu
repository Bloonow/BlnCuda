#include <cstdio>
#include <cstdint>

template <int block_size>
__global__ void flush_l2_kernel(const int* src, int* dst) {
    const int* warp_ptr = src + blockIdx.x * block_size + threadIdx.x / 32 * 32;
    const int lane_id = threadIdx.x % 32;

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int *ldg_ptr = warp_ptr + (lane_id ^ i);
        asm volatile (
            "{\n"
            ".reg .s32 val;\n"
            "ld.global.cg.b32 val, [%1];\n"
            "add.s32 %0, val, %0;\n"
            "}\n"
            : "+r"(sum)
            : "l"(ldg_ptr)
        );
    }

    if (sum != 0) {
        *dst = sum;  // 防止编译器优化
    }
}

void flush_l2() {
    const int block_size = 128;
    const int l2_flush_bytes = (1 << 20) * 128;  // 用于刷新L2缓存的数据，128MiB
    const int l2_flush_num = l2_flush_bytes / sizeof(int);
    int *src, *dst;
    cudaMalloc(&src, l2_flush_bytes);
    cudaMalloc(&dst, sizeof(int));
    cudaMemset(src, 0, l2_flush_bytes);

    flush_l2_kernel<block_size><<<l2_flush_num / block_size, block_size>>>(src, dst);

    cudaFree(src);
    cudaFree(dst);
}

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
__global__ void dram_latency_kernel(const uint32_t* src, uint32_t* dst, uint32_t* clk) {
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
    for (int i = 0; i < round; ++i) {
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
    const int stride = 1024;  // 一个线程的两条相邻ldg指令所读取内存地址之间的跨步，应大于L2缓存行
    const int round = 10;     // 执行多少次ldg指令
    const int data_bytes = stride * (round + 1);
    static_assert(stride >= 32 * sizeof(uint32_t) && stride % sizeof(uint32_t) == 0, "'stride' is invalid");

    uint32_t *src, *dst;
    uint32_t *clk;
    cudaMalloc(&src, data_bytes);
    cudaMalloc(&dst, sizeof(uint32_t));
    cudaMalloc(&clk, 32 * sizeof(uint32_t));
    uint32_t *h_src;
    cudaMallocHost(&h_src, data_bytes);
    for (int i = 0; i < data_bytes / sizeof(uint32_t); i++) h_src[i] = stride;
    cudaMemcpy(src, h_src, data_bytes, cudaMemcpyHostToDevice);

    // 预热，占用L0指令缓存和L1指令缓存
    dram_latency_kernel<round><<<1, 32>>>(src, dst, clk);
    // 刷新L2缓存
    flush_l2();
    // 测试全局内存延迟
    dram_latency_kernel<round><<<1, 32>>>(src, dst, clk);

    uint32_t h_clk[32];
    cudaMemcpy(h_clk, clk, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("DRAM latency: %u cycles\n", h_clk[0] / round);

    cudaFree(src);
    cudaFree(dst);
    cudaFree(clk);
    cudaFreeHost(h_src);
    return 0;
}