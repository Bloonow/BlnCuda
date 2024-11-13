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

template <int block_size, int round>
__global__ void smem_bandwidth_kernel(uint32_t* dst, uint32_t* clk_start, uint32_t* clk_stop) {
    __shared__ uint4 smem_buf[block_size + round];
    const uint32_t tid = threadIdx.x;
    const uint32_t smem_addr = cvt_smem_addr(smem_buf + tid);
    uint4 reg = make_uint4(tid, tid + 1, tid + 2, tid + 3);  // 用于测试共享内存的数据

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
        // 向共享内存中写入uint4类型的数据
        asm volatile (
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(smem_addr + i * (uint32_t)sizeof(uint4)), "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w)
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
    if (threadIdx.x % 32 == 0) {
        // 由每个Warp的0号线程记录时钟
        clk_start[threadIdx.x / 32] = start;
        clk_stop[threadIdx.x / 32] = stop;
    }

    int tmp = reinterpret_cast<int*>(smem_buf)[tid];
    if (tmp == -1) {
        *dst = tmp;  // 防止编译器优化
    }
}

int main(int argc, char* argv[]) {
    const int block_size = 256;
    const int round = 512;  // 执行多少次sts指令
    // 因为在Maxwell及更高的GPU架构中，一个SM最多4个处理块分区，支持4个Warp并行执行
    // 故至少需要4*32=128个线程，来充分利用一个SM当中所有处理块分区的LSU部件
    static_assert(block_size >= 128, "block_size is not enough to utilze all LSU");

    uint32_t *dst;
    uint32_t *clk_start;
    uint32_t *clk_stop;
    cudaMalloc(&dst, sizeof(uint32_t));
    cudaMalloc(&clk_start, block_size / 32 * sizeof(uint32_t));
    cudaMalloc(&clk_stop, block_size / 32 * sizeof(uint32_t));

    const int warmup_amount = 100;
    // 预热，占用L0指令缓存和L1指令缓存
    for (int i = 0; i < warmup_amount; i++) {
        smem_bandwidth_kernel<block_size, round><<<1, block_size>>>(dst, clk_start, clk_stop);
    }
    // 测试共享内存带宽
    smem_bandwidth_kernel<block_size, round><<<1, block_size>>>(dst, clk_start, clk_stop);

    uint32_t h_clk_start[block_size / 32];
    uint32_t h_clk_stop[block_size / 32];
    cudaMemcpy(h_clk_start, clk_start, block_size / 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clk_stop, clk_stop, block_size / 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t start_min = ~0;
    uint32_t stop_max = 0;
    for (int i = 0; i < block_size / 32; i++) {
        start_min = min(start_min, h_clk_start[i]);
        stop_max = max(stop_max, h_clk_stop[i]);
    }
    uint32_t duration = stop_max - start_min;
    uint32_t smem_bytes = block_size * round * sizeof(uint4);
    double bw_measured = double(smem_bytes) / duration;
    uint32_t bw_theoretical = ((uint32_t)bw_measured + 31) / 32 * 32;

    printf("Shared memory accessed: %u byte\n", smem_bytes);
    printf("Duration: %u cycle\n", duration);
    printf("Shared memory bandwidth per SM (measured): %.2f Byte/cycle\n", bw_measured);
    printf("Shared memory bandwidth per SM (theoretical): %u Byte/cycle\n", bw_theoretical);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t clk = prop.clockRate / 1000;  // KHz / 1000 = MHz
    uint32_t sm = prop.multiProcessorCount;
    double chip_bandwidth = double(sm) * bw_theoretical * clk / 1024;  // GiB/s
    printf("Standard clock frequency: %u MHz\n", clk);
    printf("SM: %u\n", sm);
    printf("Whole chip shared memory bandwidth (theoretical): %.2f GiB/s\n", chip_bandwidth);

    cudaFree(dst);
    cudaFree(clk_start);
    cudaFree(clk_stop);
    return 0;
}