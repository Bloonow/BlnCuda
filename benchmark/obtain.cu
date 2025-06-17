/**
 * 用于测试GPU的设备内存（dram）、共享内存（smem）、L2缓存（l2cache）、L1缓存（l1cache）的参数，
 * 包括延迟（latency）和带宽（bandwidth），支持Maxwell架构及之上的GPU设备。
 * 
 * 为获得最好的测试结果，在编译时应指定-arch=sm_XY编译选项，也可使用-Xptxas=-v查看寄存器和常量内存信息。
 * 例如，在NVIDIA GeForce RTX 3090 GPU设备上，测试设备内存的带宽，可以按照如下形式进行编译。
 * 
 * `nvcc -arch=sm_86 -Xptxas=-v benchmark.cu -o run`
 * 
 */

#include <cstdio>
#include <cuda.h>
#include "latency.cu"
#include "bandwidth.cu"

int main(int argc, char* argv[]) {
    benchmark::Latency latency;
    benchmark::Bandwidth bandwidth;
    printf("DRAM    latency: %4u cycle\n", latency.dram_latency);
    printf("SMEM    latency: %4u cycle\n", latency.smem_latency);
    printf("L2Cache latency: %4u cycle\n", latency.l2cache_latency);
    printf("L1Cache latency: %4u cycle\n", latency.l1cache_latency);
    printf("Dram    Bandwidth: %8.2f GiB/s\n", bandwidth.dram_read_GiB_per_second);
    printf("Smem    Bandwidth: %8.2f GiB/s\n", bandwidth.smsm_GiB_per_second);
    printf("L2Cache Bandwidth: %8.2f GiB/s\n", bandwidth.l2cache_GiB_per_second);
    return 0;
}