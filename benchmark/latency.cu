/**
 * 测试GPU设备各级存储器的访存延迟。若要准确测试存储器的延迟，有几个关键问题需要解决。
 *
 * 一、如何控制代码层面中的一条访存指令，使其在硬件层面上，确实是准确落到所测试的存储器上？
 *
 *   (1) 对于全局内存，只要确保在读取数据时，L2缓存中全是无效数据即可。
 *       假设L2缓存的总容量是N个字节，则在开始测试之前，读取大于等于N个字节的无用数据即可。
 *   (2) 对于共享内存，正常使用访问共享内存存储状态空间的访存指令即可。
 *   (3) 对于L2缓存，要确保在读取数据时，所访问的数据都在L2中存在缓存项，即所有访存都命中L2缓存。
 *       即在开始测试之前，先读取几遍数据，以确保系统将所有数据都放置到L2缓存中。
 *   (4) 对于L1缓存，与L2缓存采取类似的处理操作。
 *
 *
 * 二、为获得更准确的测试结果，通常会连续发射多条访存指令，并取这些指令的平均延迟作为测试结果。
 *     而对于现代编译器和驱动程序而言，多条连续的访存指令如果访问连续的存储位置，则会合并为一次访存请求，从而导致测试误差。
 *     如何避免编译器和驱动程序的自动优化，避免多条访存指令被合并为一次访存请求？
 *
 *   多条访存指令被合并为一次访存请求的前提条件有两个：
 *   一是硬件设备的数据总线支持一次性访问一个内存块（通常比一条访存指令所访问的数据量要大）。
 *   二是多条访存指令之间没有依赖，即后一条指令的执行，不依赖于前一条指令的结果。
 *   于是，显而易见，若想要避免访存指令被合并，最简单直接的方式就是，后一条访存指令的访问地址，需要依赖于前一条访存指令的结果。
 *   也即，前一条指令访问得到的数据，实际上是下一条指令要访问的内存地址，或是下一条指令要访问的内存地址的偏移量。
 *
 *
 * 三、指令在执行时，可能需要从设备局存中加载指令，也可能需要从L1指令缓存中加载指令。如何减少指令发射引入的误差？
 *
 *   该问题无法彻底避免，只能通过在正式测试之前，多执行几遍相同的指令，以期望将指令缓存到L1指令Cache当中。
 */

#pragma once

#include <cuda.h>

namespace benchmark {

namespace dram_latency {
__global__ void flush_l2cache_kernel(const uint32_t* array, uint32_t* dummy) {
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const uint32_t* warp_ptr = array + blockIdx.x * blockDim.x + warp_idx * 32;
    uint32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const uint32_t* ldg_ptr = warp_ptr + (lane_idx ^ i);  // 蝶形交错
        asm volatile (
            "{\n"
            ".reg.u32 val;\n"
            "ld.global.cg.b32 val, [%1];\n"
            "add.u32 %0, val, %0;\n"
            "}\n"
            : "+r"(sum)
            : "l"(ldg_ptr)
        );
    }
    dummy[threadIdx.x] = sum;  // 防止编译器优化
}

__host__ void flush_l2cache() {
    const size_t l2cache_flush_bytes = (1u << 20) * 128;  // 用于刷新L2缓存的数据，128MiB
    const size_t l2cache_flush_data_num = l2cache_flush_bytes / sizeof(uint32_t);
    uint32_t *array, *dummy;
    cudaMalloc(&array, l2cache_flush_bytes);
    cudaMalloc(&dummy, 128 * sizeof(uint32_t));
    cudaMemset(array, 0, l2cache_flush_bytes);
    flush_l2cache_kernel<<<l2cache_flush_data_num / 128, 128>>>(array, dummy);
    cudaFree(array);
    cudaFree(dummy);
}

__device__ __forceinline__ uint32_t ldg_cg(const void* ldg_ptr) {
    uint32_t value;
    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(value)
        : "l"(ldg_ptr)
        : "memory"
    );
    return value;
}

template <int Round>
__launch_bounds__(32, 1)  /* 一个SM执行最多1个线程块，一个线程块最多32个线程 */
__global__ void dram_latency_kernel(const uint32_t* array, uint32_t* dummy, uint32_t* clock) {
    uint32_t value;
    const uint32_t* ldg_ptr = array + threadIdx.x;

    // 下一次访问的地址，依赖于当前访问读取的值；值得注意的是，IADD延迟远小于要测试的延迟，可以忽略不记
    value = ldg_cg(ldg_ptr);
    ldg_ptr += value;

    uint32_t start, stop;  // 获取GPU运行时的时钟计数
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < Round; ++i) {
        // 下一次访问的地址，依赖于当前访问读取的值；值得注意的是，IADD延迟远小于要测试的延迟，可以忽略不记
        value = ldg_cg(ldg_ptr);
        ldg_ptr += value;
    }
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory"
    );
    clock[threadIdx.x] = (stop - start) / Round;

    if (value == 0) {
        dummy[threadIdx.x] = value;  // 防止编译器优化
    }
}

template <int Round = 100, int Warmup = 100>
__host__ uint32_t dram_latency() {
    // 一个线程块中32个线程，一个线程一次访问一个4字节的uint32_t类型元素，于是，一个线程块一次性访问128字节
    // 一个线程的两个相邻的ldg指令所访问的地址之间的间隔，应该大于L2Cache的缓存行（128字节）
    const size_t stride_bytes = 1024u;
    const size_t stride_num = stride_bytes / sizeof(uint32_t);
    const size_t data_bytes = (Round + 1) * stride_bytes;
    const size_t data_num = data_bytes / sizeof(uint32_t);
    static_assert(stride_bytes >= 32 * sizeof(uint32_t) && stride_bytes % sizeof(uint32_t) == 0, "stride_bytes is invalid");

    uint32_t *array, *dummy, *clock;
    cudaMalloc(&array, data_bytes);
    cudaMalloc(&dummy, 32 * sizeof(uint32_t));
    cudaMalloc(&clock, 32 * sizeof(uint32_t));
    uint32_t *h_array;
    cudaMallocHost(&h_array, data_bytes);
    for (int i = 0; i < data_num; ++i) {
        h_array[i] = stride_num;
    }
    cudaMemcpy(array, h_array, data_bytes, cudaMemcpyHostToDevice);

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = 0; i < Warmup; ++i) {
        dram_latency_kernel<Round><<<1, 32>>>(array, dummy, clock);
    }
    // 刷新L2缓存
    flush_l2cache();
    // 测试设备全局内存的延迟
    dram_latency_kernel<Round><<<1, 32>>>(array, dummy, clock);

    uint32_t h_clock[32];
    cudaMemcpy(h_clock, clock, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFreeHost(h_array);
    cudaFree(array);
    cudaFree(dummy);
    cudaFree(clock);
    return h_clock[0];
}

}  // namespace dram_latency

namespace smem_latency {
__device__ __forceinline__ uint32_t cvta_smem_addr(const void* smem_ptr) {
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

template <int Round>
__launch_bounds__(16, 1)  /* 一个SM执行最多1个线程块，一个线程块最多16个线程 */
__global__ void smem_latency_kernel(const uint32_t* array, uint32_t* dummy, uint32_t* clock) {
    __shared__ uint32_t smem_buffer[16];
    uint32_t smem_addr = cvta_smem_addr(smem_buffer + threadIdx.x);  // 使用.shared存储状态空间中的相对地址
    smem_buffer[threadIdx.x] = array[threadIdx.x];                   // smem_buffer[i]存储的是自己本身的地址
    __syncthreads();

    uint32_t start, stop;
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < Round; ++i) {
        // 下一次访问的地址，依赖于当前访问读取的值
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
    clock[threadIdx.x] = (stop - start) / Round;

    if (smem_addr == 0) {
        dummy[threadIdx.x] = smem_addr;  // 防止编译器优化
    }
}

template <int Round = 100, int Warmup = 100>
__host__ uint32_t smem_latency() {
    // 一个线程块中16个线程，一个线程一次访问一个4字节的uint32_t类型元素，于是，一个线程块一次性访问64字节
    uint32_t *array, *dummy, *clock;
    cudaMalloc(&array, 16 * sizeof(uint32_t));
    cudaMalloc(&dummy, 16 * sizeof(uint32_t));
    cudaMalloc(&clock, 16 * sizeof(uint32_t));
    uint32_t *h_array;
    cudaMallocHost(&h_array, 16 * sizeof(uint32_t));
    for (int i = 0; i < 16; ++i) {
        h_array[i] = i * sizeof(uint32_t);
    }
    cudaMemcpy(array, h_array, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = 0; i < Warmup; ++i) {
        smem_latency_kernel<Round><<<1, 16>>>(array, dummy, clock);
    }
    // 测试共享内存的延迟
    smem_latency_kernel<Round><<<1, 16>>>(array, dummy, clock);

    uint32_t h_clock[16];
    cudaMemcpy(h_clock, clock, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFreeHost(h_array);
    cudaFree(array);
    cudaFree(dummy);
    cudaFree(clock);
    return h_clock[0];
}

}  // namespace smem_latency

namespace l2cache_latency {
__device__ __forceinline__ uint32_t ldg_cg(const void* ldg_ptr) {
    uint32_t value;
    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(value)
        : "l"(ldg_ptr)
        : "memory"
    );
    return value;
}

template <int Round>
__launch_bounds__(32, 1)  /* 一个SM执行最多1个线程块，一个线程块最多32个线程 */
__global__ void l2cache_latency_kernel(const uint32_t* array, uint32_t* dummy, uint32_t* clock) {
    uint32_t value;
    const uint32_t* ldg_ptr = array + threadIdx.x;

    // 下一次访问的地址，依赖于当前访问读取的值；值得注意的是，IADD延迟远小于要测试的延迟，可以忽略不记
    value = ldg_cg(ldg_ptr);
    ldg_ptr += value;

    uint32_t start, stop;
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < Round; ++i) {
        // 下一次访问的地址，依赖于当前访问读取的值；值得注意的是，IADD延迟远小于要测试的延迟，可以忽略不记
        value = ldg_cg(ldg_ptr);
        ldg_ptr += value;
    }
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(stop)
        :
        : "memory"
    );
    clock[threadIdx.x] = (stop - start) / Round;

    if (value == 0) {
        dummy[threadIdx.x] = value;  // 防止编译器优化
    }
}

template <int Round = 100, int Warmup = 100>
__host__ uint32_t l2cache_latency() {
    // 一个线程块中32个线程，一个线程一次访问一个4字节的uint32_t类型元素，于是，一个线程块一次性访问128字节
    // 一个线程的两个相邻的ldg指令所访问的地址之间的间隔，应该大于L2Cache的缓存行（128字节）
    const size_t stride_bytes = 128u;
    const size_t stride_num = stride_bytes / sizeof(uint32_t);
    const size_t data_bytes = (Round + 1) * stride_bytes;
    const size_t data_num = data_bytes / sizeof(uint32_t);
    static_assert(stride_bytes >= 32 * sizeof(uint32_t) && stride_bytes % sizeof(uint32_t) == 0, "stride_bytes is invalid");

    uint32_t *array, *dummy, *clock;
    cudaMalloc(&array, data_bytes);
    cudaMalloc(&dummy, 32 * sizeof(uint32_t));
    cudaMalloc(&clock, 32 * sizeof(uint32_t));
    uint32_t *h_array;
    cudaMallocHost(&h_array, data_bytes);
    for (int i = 0; i < data_num; ++i) {
        h_array[i] = stride_num;
    }
    cudaMemcpy(array, h_array, data_bytes, cudaMemcpyHostToDevice);

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = 0; i < Warmup; ++i) {
        l2cache_latency_kernel<Round><<<1, 32>>>(array, dummy, clock);
    }
    // 测试L2缓存的延迟
    l2cache_latency_kernel<Round><<<1, 32>>>(array, dummy, clock);

    uint32_t h_clock[32];
    cudaMemcpy(h_clock, clock, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFreeHost(h_array);
    cudaFree(array);
    cudaFree(dummy);
    cudaFree(clock);
    return h_clock[0];
}

}  // namespace l2cache_latency

namespace l1cache_latency {
using ptr_t = void*;

template <int Round>
__launch_bounds__(4, 1)  /* 一个SM执行最多1个线程块，一个线程块最多4个线程 */
__global__ void l1cache_latency_kernel(ptr_t* array, ptr_t* dummy, uint32_t* clock) {
    ptr_t* ldg_ptr = array + threadIdx.x;

    // 下一次访问的地址，依赖于当前访问读取的值
    for (int i = 0; i < Round; ++i) {
        asm volatile (
            "ld.global.nc.b64 %0, [%0];\n"
            : "+l"(ldg_ptr)
            :
            : "memory"
        );
    }

    uint32_t start, stop;
    asm volatile (
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < Round; ++i) {
        // 下一次访问的地址，依赖于当前访问读取的值
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
    clock[threadIdx.x] = (stop - start) / Round;

    if (ldg_ptr == nullptr) {
        dummy[threadIdx.x] = ldg_ptr;  // 防止编译器优化
    }
}

template <int Round = 100, int Warmup = 100>
__host__ uint32_t l1cache_latency() {
    // 一个线程块中4个线程，一个线程一次访问一个8字节的ptr_t类型元素，于是，一个线程块一次性访问32字节
    ptr_t *array, *dummy;
    uint32_t* clock;
    cudaMalloc(&array, 4 * sizeof(ptr_t));
    cudaMalloc(&dummy, 4 * sizeof(ptr_t));
    cudaMalloc(&clock, 4 * sizeof(uint32_t));
    ptr_t *h_array;
    cudaMallocHost(&h_array, 4 * sizeof(ptr_t));
    for (int i = 0; i < 4; ++i) {
        h_array[i] = array + i;
    }
    cudaMemcpy(array, h_array, 4 * sizeof(ptr_t), cudaMemcpyHostToDevice);  // array[i]存储的是自己本身的地址

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = 0; i < Warmup; ++i) {
        l1cache_latency_kernel<Round><<<1, 4>>>(array, dummy, clock);
    }
    // 测试L1缓存的延迟
    l1cache_latency_kernel<Round><<<1, 4>>>(array, dummy, clock);

    uint32_t h_clock[4];
    cudaMemcpy(h_clock, clock, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFreeHost(h_array);
    cudaFree(array);
    cudaFree(dummy);
    cudaFree(clock);
    return h_clock[0];
}

}  // namespace l1cache_latency

struct Latency {
    uint32_t dram_latency;
    uint32_t smem_latency;
    uint32_t l2cache_latency;
    uint32_t l1cache_latency;
    Latency() { this->obtain(); }

    void obtain() {
        dram_latency = benchmark::dram_latency::dram_latency();
        smem_latency = benchmark::smem_latency::smem_latency();
        l2cache_latency = benchmark::l2cache_latency::l2cache_latency();
        l1cache_latency = benchmark::l1cache_latency::l1cache_latency();
    }
};

}  // namespace benchmark