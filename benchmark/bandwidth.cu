/**
 * 测试GPU设备各级存储器的访存带宽。若要准确测试存储器的带宽，有几个关键问题需要解决。
 *
 * 一、如何控制代码层面中的一条访存指令，使其在硬件层面上，确实是准确落到所测试的存储器上？
 *
 *   (1) 对于全局内存，只要确保在读取数据时，L2缓存中全是无效数据即可。
 *       假设L2缓存的总容量是N个字节，则在开始测试之前，读取大于等于N个字节的无用数据即可。
 *   (2) 对于共享内存，正常使用访问共享内存存储状态空间的访存指令即可。
 *   (3) 对于L2缓存，要确保在读取数据时，所访问的数据都在L2中存在缓存项，即所有访存都命中L2缓存。
 *       即在开始测试之前，先读取几遍数据，以确保系统将所有数据都放置到L2缓存中。
 *
 *
 * 二、考虑硬件结构，存储器依赖数据总线来传输数据，并且同一时刻传递的数据量取决于数据总线的宽度。
 *     在测试存储器的带宽时，如何确保在硬件上的数据总线是全部被激活并占用了？
 *
 *   通常而言，注意设备全局内存的合并访存，注意共享内存的向量化访问，注意L2缓存的128字节的缓存行，即可。
 *
 *
 * 三、指令在执行时，可能需要从设备局存中加载指令，也可能需要从L1指令缓存中加载指令。如何减少指令发射引入的误差？
 *
 *   该问题无法彻底避免，只能通过在正式测试之前，多执行几遍相同的指令，以期望将指令缓存到L1指令Cache当中。
 */

 #pragma once

#include <cuda.h>

namespace benchmark {

namespace dram_bandwidth {
struct Result {
    size_t data_bytes;
    double read_GiB_per_second;
    double write_GiB_per_second;
    double copy_GiB_per_second;
    Result(size_t data_bytes_ = 0) : data_bytes(data_bytes_) {}
};

__device__ __forceinline__ uint4 ldg_cs(const void* ldg_ptr) {
    uint4 value;
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(ldg_ptr)
        : "memory"
    );
    return value;
}

__device__ __forceinline__ void stg_cs(const uint4& value, void* stg_ptr) {
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        :
        : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(stg_ptr)
        : "memory"
    );
}

template <int ThreadTile = 1>
__global__ void dram_read_kernel(const void* array, void* dummy) {
    const uint4* ldg_ptr = reinterpret_cast<const uint4*>(array) + blockIdx.x * blockDim.x * ThreadTile + threadIdx.x;
    uint4 value[ThreadTile];
    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        value[i] = ldg_cs(ldg_ptr + i * blockDim.x);  // 读设备的全局内存，并避免使用L2缓存
    }
    uint4* dummy_ptr = reinterpret_cast<uint4*>(dummy) + blockIdx.x * blockDim.x * ThreadTile + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        // 防止编译器优化，为尽量避免该代码块的影响，初始化时应将数组array全部赋为0值
        if (value[i].x != 0) {
            stg_cs(value[i], dummy_ptr + i * blockDim.x);
        }
    }
}

template <int ThreadTile = 1>
__global__ void dram_write_kernel(void* destination) {
    uint4* stg_ptr = reinterpret_cast<uint4*>(destination) + blockIdx.x * blockDim.x * ThreadTile + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        stg_cs(make_uint4(i, i, i, i), stg_ptr + i * blockDim.x);
    }
}

template <int ThreadTile = 1>
__global__ void dram_copy_kernel(const void* array, void* destination) {
    const uint4* ldg_ptr = reinterpret_cast<const uint4*>(array) + blockIdx.x * blockDim.x * ThreadTile + threadIdx.x;
    uint4* stg_ptr = reinterpret_cast<uint4*>(destination) + blockIdx.x * blockDim.x * ThreadTile + threadIdx.x;
    uint4 value[ThreadTile];
    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        value[i] = ldg_cs(ldg_ptr + i * blockDim.x);
    }
    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        stg_cs(value[i], stg_ptr + i * blockDim.x);
    }
}

template <int Round = 100, int Warmup = 100, int ThreadTile = 1, size_t skew_bytes = (1ull << 20) * 16>
__host__ Result dram_bandwidth(const size_t data_bytes = (1ull << 30) * 2) {
    // 一个线程块128个线程，一个线程一共处理ThreadTile个类型为uint4的元素，一共由若干个线程块处理data_bytes字节的数据
    Result result(data_bytes);
    const size_t data_num = data_bytes / sizeof(uint4);
    const size_t grid_size = data_num / (128 * ThreadTile);
    static_assert(Round >= Warmup, "Round should be greater than or equal to Warmup");
    static_assert(skew_bytes % sizeof(uint4) == 0, "skew_bytes is invalid");

    unsigned char* workspace;
    cudaMalloc(&workspace, data_bytes + Round * skew_bytes);
    cudaMemset(workspace, 0, data_bytes + Round * skew_bytes);

    // 为保证准确性，每轮测试时都将数据起始位置进行一定偏移，以skew_bytes指定偏移的字节数
    // 在下面的代码中，循环采用从Round-1到0的反向迭代，而不是采用从0到Round-1的正向迭代，是有深刻原因的，这与L2缓存项的替换规则有关
    // (1) 若采用正向迭代，在前一轮结束之后，下一轮所访问的第一块数据，会直接命中Cache，因为上一轮已经访问过
    // (2) 若采用反向迭代，在前一轮结束之后，下一轮所访问的第一块数据，总是不会命中Cache，因为上一轮没有访问过，从而避免L2的影响；
    //   而且，在下一轮访问第一块未命中Cache的数据时，总会把所要访问的下一块数据从Cache中替换出去，因为下一块数据是上一轮中最早访问的

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = Warmup - 1; i >= 0; --i) {
        dram_read_kernel<ThreadTile><<<grid_size, 128>>>(workspace + i * skew_bytes, workspace);
        dram_write_kernel<ThreadTile><<<grid_size, 128>>>(workspace + i * skew_bytes);
        dram_copy_kernel<ThreadTile><<<grid_size / 2, 128>>>(workspace + i * skew_bytes, workspace + i * skew_bytes + data_bytes / 2);
    }

    float time_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 读全局内存
    cudaEventRecord(start);
    for (int i = Round - 1; i >= 0; --i) {
        dram_read_kernel<ThreadTile><<<grid_size, 128>>>(workspace + i * skew_bytes, workspace);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    result.read_GiB_per_second = (static_cast<double>(data_bytes) * Round / (1u << 30)) / (static_cast<double>(time_ms) / 1000);

    // 写全局内存
    cudaEventRecord(start);
    for (int i = Round - 1; i >= 0; --i) {
        dram_write_kernel<ThreadTile><<<grid_size, 128>>>(workspace + i * skew_bytes);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    result.write_GiB_per_second = (static_cast<double>(data_bytes) * Round / (1u << 30)) / (static_cast<double>(time_ms) / 1000);

    // 复制全局内存
    cudaEventRecord(start);
    for (int i = Round - 1; i >= 0; --i) {
        dram_copy_kernel<ThreadTile><<<grid_size / 2, 128>>>(workspace + i * skew_bytes, workspace + i * skew_bytes + data_bytes / 2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    result.copy_GiB_per_second = (static_cast<double>(data_bytes) * Round / (1u << 30)) / (static_cast<double>(time_ms) / 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(workspace);
    return result;
}

}  // namespace dram_bandwidth

namespace smem_bandwidth {
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

template <int Round = 512>
__global__ void smem_bandwidth_kernel(int* dummy, uint32_t* clock_start, uint32_t* clock_stop) {
    extern __shared__ uint4 smem_buffer[];
    const uint32_t smem_addr = cvta_smem_addr(smem_buffer + threadIdx.x);  // 使用.shared存储状态空间中的相对地址
    uint4 value = make_uint4(threadIdx.x, threadIdx.x, threadIdx.x, threadIdx.x);

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
        asm volatile (
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "r"(smem_addr + i * (uint32_t)sizeof(uint4)), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
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
        // 对于共享内存访问操作，一个Warp中的32个线程是同步执行的，故由Warp中的0号线程记录时钟
        clock_start[threadIdx.x / 32] = start;
        clock_stop[threadIdx.x / 32] = stop;
    }

    int temp = reinterpret_cast<int*>(smem_buffer)[threadIdx.x];
    if (temp == -1) {
        dummy[threadIdx.x] = temp;  // 防止编译器优化
    }
}

template <int Round = 512, int Warmup = 100>
__host__ double smem_bandwidth() {
    // 一个线程块中512个线程，一个线程一次访问一个16字节的uint4类型元素
    // 因为在Maxwell及更高的GPU架构中，一个SM最多4个处理块分区，支持4个Warp并行执行
    // 故至少需要4 * 32 = 128个线程，来充分利用一个SM当中所有处理块分区的LSU部件
    const int block_size = 512;
    const int warp_num = block_size / 32;
    const size_t smem_bytes = (block_size + Round) * sizeof(uint4);
    static_assert(Round >= Warmup, "Round should be greater than or equal to Warmup");

    int *dummy;
    uint32_t *clock_start, *clock_stop;
    cudaMalloc(&dummy, block_size * sizeof(uint32_t));
    cudaMalloc(&clock_start, warp_num * sizeof(uint32_t));
    cudaMalloc(&clock_stop, warp_num * sizeof(uint32_t));

    // 预热，以期望将指令缓存到L1指令Cache当中
    for (int i = 0; i < Warmup; ++i) {
        smem_bandwidth_kernel<Round><<<1, block_size, smem_bytes>>>(dummy, clock_start, clock_stop);
    }
    // 测试共享内存带宽
    smem_bandwidth_kernel<Round><<<1, block_size, smem_bytes>>>(dummy, clock_start, clock_stop);

    uint32_t h_clock_start[warp_num], h_clock_stop[warp_num];
    cudaMemcpy(h_clock_start, clock_start, warp_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clock_stop, clock_stop, warp_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t start = ~0, stop = 0;
    for (int i = 0; i < warp_num; ++i) {
        start = min(start, h_clock_start[i]);
        stop = max(stop, h_clock_stop[i]);
    }
    uint32_t clock_duration = stop - start;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t clock = prop.clockRate * 1000;  // KHz * 1000 = Hz
    uint32_t sm = prop.multiProcessorCount;
    size_t data_bytes = block_size * Round * sizeof(uint4);
    double bw_measured = static_cast<double>(data_bytes) / clock_duration;      // Byte/cycle per sm
    double bw_theoretical = (static_cast<size_t>(bw_measured) + 31) / 32 * 32;  // Byte/cycle per sm
    double GiB_per_second = static_cast<double>(sm) * clock * bw_theoretical / (1ull << 30);

    cudaFree(dummy);
    cudaFree(clock_start);
    cudaFree(clock_stop);
    return GiB_per_second;
}

}  // namespace smem_bandwidth

namespace l2cache_bandwidth {
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

template <int ThreadTile = 16>
__global__ void l2cache_bandwidth_kernel(const uint32_t* array, uint32_t* dummy, const size_t data_nums) {
    const uint32_t* ldg_ptr = array + ((blockIdx.x * blockDim.x * ThreadTile + threadIdx.x) % data_nums);
    uint32_t value[ThreadTile];

    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        value[i] = ldg_cg(ldg_ptr + i * blockDim.x);
    }

    #pragma unroll
    for (int i = 0; i < ThreadTile; ++i) {
        if (value[i] != 0) {
            dummy[threadIdx.x] = value[i];  // 防止编译器优化
        }
    }
}

template <int Round = 100, int Warmup = 100, int ThreadTile = 16>
__host__ double l2cache_bandwidth() {
    // 一个线程块中128个线程，一个线程一次访问一个4字节的uint32_t类型元素
    const size_t data_bytes = (1u << 20) * 4;               // 数据的字节数，需要小于L2缓存的容量
    const size_t data_num = data_bytes / sizeof(uint32_t);  // 数据的数目
    const size_t ldg_num = (1u << 20) * 512;  // 总的ldg指令的执行次数，为测试准确，大于数据的数目，因此，数据会被重复读取
    const size_t grid_size = ldg_num / (128 * ThreadTile);
    static_assert(data_num >= 128 * ThreadTile && data_num % (128 * ThreadTile) == 0, "data_num is invalid");

    uint32_t *array, *dummy;
    cudaMalloc(&array, data_bytes);
    cudaMalloc(&dummy, 128 * sizeof(uint32_t));
    cudaMemset(array, 0, data_bytes);

    float time_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热，以期望将指令缓存到L1指令Cache当中，并将数据缓存到L2缓存中
    for (int i = 0; i < Warmup; ++i) {
        l2cache_bandwidth_kernel<<<grid_size, 128>>>(array, dummy, data_num);
    }
    cudaEventRecord(start);
    for (int i = 0; i < Round; ++i) {
        l2cache_bandwidth_kernel<<<grid_size, 128>>>(array, dummy, data_num);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    double GiB_per_second = (static_cast<double>(ldg_num * sizeof(uint32_t)) / (1u << 30)) / (static_cast<double>(time_ms) / 1000 / Round);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(array);
    cudaFree(dummy);
    return GiB_per_second;
}

}  // namespace l2cache_bandwidth

struct Bandwidth {
    double dram_read_GiB_per_second;
    double dram_write_GiB_per_second;
    double dram_copy_GiB_per_second;
    double smsm_GiB_per_second;
    double l2cache_GiB_per_second;
    Bandwidth() { this->obtain(); }

    void obtain() {
        benchmark::dram_bandwidth::Result dram_bw = benchmark::dram_bandwidth::dram_bandwidth();
        dram_read_GiB_per_second = dram_bw.read_GiB_per_second;
        dram_write_GiB_per_second = dram_bw.write_GiB_per_second;
        dram_copy_GiB_per_second = dram_bw.copy_GiB_per_second;
        smsm_GiB_per_second = benchmark::smem_bandwidth::smem_bandwidth();
        l2cache_GiB_per_second = benchmark::l2cache_bandwidth::l2cache_bandwidth();
    }
};

}  // namespace benchmark