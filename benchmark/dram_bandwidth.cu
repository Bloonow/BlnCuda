#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint4 ldg_cs(const void* ptr) {
    uint4 ret;
    // ld.global指令从设备全局内存中进行读取
    // cs适用于流式（streaming）数据，指示程序以驱逐优先（evict-first）策略分配L1或L2缓存行，这种流数据可能只被访问一两次
    // cs策略可以避免缓存污染（cache pollution），即避免缓存中存储大量不必要的数据
    // b32表示将数据的类型解析为32位的Bits(untyped)
    // memory表示该内联汇编会影响内存状态，编译器应该避免优化掉相关内存操作，这在循环或并行执行时尤为重要
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr)
        : "memory"
    );
    return ret;
}

__device__ __forceinline__ void stg_cs(const uint4& reg, void* ptr) {
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        :
        : "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr)
        : "memory"
    );
}

template <int block_size, int thread_tile>
__global__ void read_kernel(const void* src, void* dst) {
    const int block_tile = block_size * thread_tile;
    const int offset = block_tile * blockIdx.x + threadIdx.x;
    const uint4* ldg_ptr = reinterpret_cast<const uint4*>(src) + offset;
    uint4 reg[thread_tile];

    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        // 读设备全局内存，并避免使用L2缓存
        reg[i] = ldg_cs(ldg_ptr + block_size * i);
    }

    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        if (reg[i].x != 0) {
            stg_cs(reg[i], (uint4*)dst + i);  // 防止编译器优化
        }
    }
}

template <int block_size, int thread_tile>
__global__ void write_kernel(void* dst) {
    const int block_tile = block_size * thread_tile;
    const int offset = block_tile * blockIdx.x + threadIdx.x;
    uint4* stg_ptr = reinterpret_cast<uint4*>(dst) + offset;

    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        uint4 reg = make_uint4(0, 0, 0, 0);
        stg_cs(reg, stg_ptr + block_size * i);
    }
}

template <int block_size, int thread_tile>
__global__ void copy_kernel(const void* src, void* dst) {
    const int block_tile = block_size * thread_tile;
    const int offset = block_tile * blockIdx.x + threadIdx.x;
    const uint4* ldg_ptr = reinterpret_cast<const uint4*>(src) + offset;
    uint4* stg_ptr = reinterpret_cast<uint4*>(dst) + offset;
    uint4 reg[thread_tile];

    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        reg[i] = ldg_cs(ldg_ptr + block_size * i);
    }
    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        stg_cs(reg[i], stg_ptr + block_size * i);
    }
}

template <int block_size, int thread_tile, int memory_round, int benchmark_amount>
void benchmark(const size_t data_bytes) {
    printf("%lu MiB (r+w)\n", data_bytes / (1 << 20));
    size_t data_num = data_bytes / sizeof(uint4);
    size_t grid_size = data_num / (block_size * thread_tile);
    static_assert(memory_round % sizeof(uint4) == 0, "memory_round is invalid");

    char* workspace;
    cudaMalloc(&workspace, data_bytes + memory_round * benchmark_amount);
    cudaMemset(workspace, 0, data_bytes + memory_round * benchmark_amount);

    float time_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    read_kernel<block_size, thread_tile><<<grid_size, block_size>>>(workspace, workspace);
    write_kernel<block_size, thread_tile><<<grid_size, block_size>>>(workspace);
    copy_kernel<block_size, thread_tile><<<grid_size / 2, block_size>>>(workspace, workspace + data_bytes / 2);

    // 读全局内存
    cudaEventRecord(start);
    for (int i = benchmark_amount - 1; i >= 0; i--) {
        read_kernel<block_size, thread_tile><<<grid_size, block_size>>>(workspace + i * memory_round, workspace);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Read : %.2f GiB/s\n", ((double)data_bytes * benchmark_amount / (1 << 30)) / ((double)time_ms / 1000));

    // 写全局内存
    cudaEventRecord(start);
    for (int i = benchmark_amount - 1; i >= 0; i--) {
        write_kernel<block_size, thread_tile><<<grid_size, block_size>>>(workspace + i * memory_round);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Write: %.2f GiB/s\n", ((double)data_bytes * benchmark_amount / (1 << 30)) / ((double)time_ms / 1000));

    // 复制全局内存
    cudaEventRecord(start);
    for (int i = benchmark_amount - 1; i >= 0; i--) {
        copy_kernel<block_size, thread_tile><<<grid_size / 2, block_size>>>(
            workspace + i * memory_round, workspace + i * memory_round + data_bytes / 2
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Copy : %.2f GiB/s\n", ((double)data_bytes * benchmark_amount / (1 << 30)) / ((double)time_ms / 1000));

    printf("------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(workspace);
}

int main(int argc, char* argv[]) {
    const int block_size = 128;
    const int thread_tile = 1;
    const int memory_round = (1u << 20) * 16;  // 16MiB
    const int benchmark_amount = 100;

    // 4MiB ~ 2GiB
    for (size_t bytes = (1ull << 20) * 4; bytes <= (1ull << 30) * 2; bytes *= 8) {
        benchmark<block_size, thread_tile, memory_round, benchmark_amount>(bytes);
    }
    return 0;
}