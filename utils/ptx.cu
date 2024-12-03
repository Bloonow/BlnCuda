#pragma once

#include <cuda.h>

namespace ptx {

__device__ __forceinline__
uint32_t smem_addr(const void *ptr) {
    // 共享内存指针 ptr 转换为 addr 地址
    uint32_t addr;
    asm volatile (
        "{\n"
        ".reg .u64 u64addr;\n"
        "cvta.to.shared.u64 u64addr, %1;\n"
        "cvt.u32.u64 %0, u64addr;\n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__
void lds(float &reg, const uint32_t addr) {
    asm volatile (
        "ld.shared.f32 %0, [%1];\n"
        : "=f"(reg)
        : "r"(addr)
    );
}

__device__ __forceinline__
void lds(float &r0, float &r1, float &r2, float &r3, const uint32_t addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts(const float &reg, const uint32_t addr) {
    asm volatile (
        "st.shared.f32 [%1], %0;\n"
        :
        : "f"(reg), "r"(addr)
    );
}

__device__ __forceinline__
void sts(const float &r0, const float &r1, const float &r2, const float &r3, const uint32_t addr) {
    asm volatile (
        "st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n"
        :
        : "f"(r0), "f"(r1), "f"(r2), "f"(r3), "r"(addr)
    );
}

__device__ __forceinline__
void ldg(float &reg, const void *ptr, bool guard) {
    // 当 guard 为 true 时，从全局内存 ptr 中读取 1 个 float 数据
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p ld.global.nc.L2::128B.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p ld.global.nc.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #endif
}

__device__ __forceinline__
void ldg_zero(float &reg, const void *ptr, bool guard) {
    // 当 guard 为 true 时，从全局内存 ptr 中读取 1 个 float 数据，否则置零
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@!p mov.b32 %0, 0;\n"
        "@p ld.global.nc.L2::128B.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@!p mov.b32 %0, 0;\n"
        "@p ld.global.nc.f32 %0, [%1];\n"
        "}\n"
        : "=f"(reg)
        : "l"(ptr), "r"(int(guard))
    );
    #endif
}

__device__ __forceinline__
void stg(const float &reg, void *ptr, bool guard) {
    // 当 guard 为 true 时，向全局内存 ptr 中写入 1 个 float 数据
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p st.global.f32 [%1], %0;\n"
        "}\n"
        :
        : "f"(reg), "l"(ptr), "r"((int)guard)
    );
}

// Ampere Architecture
#define __CUDA_ARCH__ 860
#if (__CUDA_ARCH__ >= 860)

__device__ __forceinline__
void ldg_sts(const void *gmem_ptr, const uint32_t smem_addr, bool guard) {
    // 当 guard 为 true 时，从全局内存 gmem_ptr 中读取数据，并写入到共享内存 smem_addr 当中，异步执行
    // 指令的第3个参数 cp_size 指定为 4 表示一次性复制 4 个字节，即 32 位
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p cp.async.ca.shared.global.L2::128B [%1], [%0], 4;\n"
        "}\n"
        :
        : "l"(gmem_ptr), "r"(smem_addr), "r"((int)guard)
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p cp.async.ca.shared.global [%1], [%0], 4;\n"
        "}\n"
        :
        : "l"(gmem_ptr), "r"(smem_addr), "r"((int)guard)
    );
    #endif
}

__device__ __forceinline__
void ldg_sts(const void *gmem_ptr, const uint32_t smem_addr, const uint32_t src_size, bool guard) {
    // 当 guard 为 true 时，从全局内存 gmem_ptr 中读取数据，并写入到共享内存 smem_addr 当中，异步执行
    // 指令的第3个参数 cp_size 指定为 4 表示一次性复制 4 个字节，即 32 位
    // 指令的第4个参数 src_size 表示要从源地址复制的字节数，目标地址剩余的 cp_size - src_size 字节数据会被置零
    #if (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4)
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p cp.async.ca.shared.global.L2::128B [%1], [%0], 4, %2;\n"
        "}\n"
        :
        : "l"(gmem_ptr), "r"(smem_addr), "r"(src_size), "r"((int)guard)
    );
    #else
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %2, 0;\n"
        "@p cp.async.ca.shared.global [%1], [%0], 4, %2;\n"
        "}\n"
        :
        : "l"(gmem_ptr), "r"(smem_addr), "r"(src_size), "r"((int)guard)
    );
    #endif
}

__device__ __forceinline__
void ldg_sts_commit() {
    // 等待异步复制指令执行完成
    asm volatile ("cp.async.wait_all;\n"::);
}

#endif

} // namespace ptx