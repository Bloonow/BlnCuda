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

template <int block_size, int thread_tile, int data_num>
__global__ void l2_bandwidth_kernel(const uint32_t* src, uint32_t* dst) {
    const int block_tile = block_size * thread_tile;
    const int offset = (block_tile * blockIdx.x + threadIdx.x) % data_num;
    const int* ldg_ptr = reinterpret_cast<const int*>(src + offset);
    int reg[thread_tile];

    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        // 读设备全局内存，确保数据已经缓存到L2当中
        reg[i] = ldg_cg(ldg_ptr + block_size * i);
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < thread_tile; i++) {
        sum += reg[i];  // 读L2缓存
    }

    if (sum != 0) {
        *dst = sum;     // 防止编译器优化
    }
}

int main(int argc, char* argv[]) {
    const int data_bytes = (1 << 20) * 2;  // 数据的字节数，2MB，需要小于L2缓存的容量
    const int data_num = data_bytes / sizeof(uint32_t);  // 数据的数目，这些数据由线程通过ldg指令读取
    const int ldg_num = (1 << 20) * 512;   // 要执行的ldg指令的总数目，一个指令读取一个数据
    const int thread_tile = 16;            // 一个线程负责的数据数目，一个数据由一个ldg指令读取
    const int block_size = 128;            // 一个线程块内的线程数目
    const int grid_size = ldg_num / block_size / thread_tile;  // 网格大小
    static_assert(
        data_num >= thread_tile * block_size && data_num % (thread_tile * block_size) == 0,
        "'thread_tile' or 'block_size' is invalid"
    );

    uint32_t *src, *dst;
    cudaMalloc(&src, data_num * sizeof(uint32_t));
    cudaMalloc(&dst, sizeof(uint32_t));
    cudaMemset(src, 0, data_num * sizeof(uint32_t));

    const int warmup_amount = 100;     // 预热次数
    const int benchmark_amount = 100;  // 测试次数
    float time_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热，将数据从设备全局内存缓存到L2当中
    for (int i = 0; i < warmup_amount; i++) {
        l2_bandwidth_kernel<block_size, thread_tile, data_num><<<grid_size, block_size>>>(src, dst);
    }
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_amount; i++) {
        l2_bandwidth_kernel<block_size, thread_tile, data_num><<<grid_size, block_size>>>(src, dst);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    double GiB_ps = ((double)(ldg_num * sizeof(uint32_t) / (1 << 30))) / ((double)(time_ms / benchmark_amount / 1000));
    printf("L2 Cache bandwidth: %.2lf GiB/s\n", GiB_ps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src);
    cudaFree(dst);
    return 0;
}