#pragma once

#include <cuda.h>
#include "gemm_utils.cu"
#include "../utils/buffer.cu"

namespace sgemm_32x32_4x4 {

using buffer::SharedMemory;

/* [WHEN] K <= 48 */
struct TileIndex {
    uint32_t brid, bcid, tid, wid, lid;
    uint32_t wrows, wcols, wrid, wcid, lrid, lcid;
    uint32_t M, N, K, aS, bS, cS;
    __device__ TileIndex(const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS) {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x; tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 2; wcid = wid % 2;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
    }
};

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 trans;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + brid * 32 * N + bcid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        trans.x = Creg[row][0]; trans.y = Creg[row][1]; trans.z = Creg[row][2]; trans.w = Creg[row][3];
        *reinterpret_cast<float4*>(
            smem_buf + (wrid * wrows * 4 * 32 + wcid * wcols * 4) + lrid * 4 * 32 + lcid * 4 + row * 32
        ) = trans;
    }
    __syncthreads();
    // 将数据从共享内存转移到全局内存
    // 使用 2x32 排列的线程搬运 32x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_row = 0; gmem_row < 32; gmem_row += 2) {
        if ((brid * 32 + gmem_row + tid / 32 < M) && (bcid * 32 + tid % 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_row + tid / 32) * N + (tid % 32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_row * 32 + tid);
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 trans;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + bcid * 32 * M + brid * 32);
    // 将所有线程的全部数据写入到共享内存
    __syncthreads();
    #pragma unroll
    for (uint32_t column = 0; column < 4; ++column) {
        trans.x = Creg[0][column]; trans.y = Creg[1][column]; trans.z = Creg[2][column]; trans.w = Creg[3][column];
        *reinterpret_cast<float4*>(
            smem_buf + (wcid * wcols * 4 * 32 + wrid * wrows * 4) + lcid * 4 * 32 + lrid * 4 + column * 32
        ) = trans;
    }
    __syncthreads();
    // 将数据从共享内存转移到全局内存
    // 使用 32x2 排列的线程搬运 32x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_column = 0; gmem_column < 32; gmem_column += 2) {
        if ((brid * 32 + tid % 32 < M) && (bcid * 32 + gmem_column + tid / 32 < N)) {
            *reinterpret_cast<float*>(
                C_block + (gmem_column + tid / 32) * M + (tid %32)
            ) = *reinterpret_cast<float*>(smem_buf + gmem_column * 32 + tid);
        }
    }
}

__device__ __forceinline__
void compute_tile_crr(
    float Creg[4][4], float *Asmem, float *Bsmem, const uint32_t &ldA, const uint32_t &ldB,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 Areg, Breg;
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        Areg = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 4 + lrid * 4 + kid * ldA);
        Breg = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 4 + lcid * 4 + kid * ldB);
        Creg[0][0] += Areg.x * Breg.x;
        Creg[0][1] += Areg.x * Breg.y;
        Creg[0][2] += Areg.x * Breg.z;
        Creg[0][3] += Areg.x * Breg.w;
        Creg[1][0] += Areg.y * Breg.x;
        Creg[1][1] += Areg.y * Breg.y;
        Creg[1][2] += Areg.y * Breg.z;
        Creg[1][3] += Areg.y * Breg.w;
        Creg[2][0] += Areg.z * Breg.x;
        Creg[2][1] += Areg.z * Breg.y;
        Creg[2][2] += Areg.z * Breg.z;
        Creg[2][3] += Areg.z * Breg.w;
        Creg[3][0] += Areg.w * Breg.x;
        Creg[3][1] += Areg.w * Breg.y;
        Creg[3][2] += Areg.w * Breg.z;
        Creg[3][3] += Areg.w * Breg.w;
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    
    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * T.N + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32) + (T.wid * 4 * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N)               B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart;
    B_tid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (256 + 32);
        Bsmem += (2 * (kth & 1) - 1) * 256;
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (256 + 32) * 2);
    
    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_tid += kstart;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (256 + 32);
        Bsmem += (2 * (kth & 1) - 1) * (256 + 32);
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    
    // [NEXT] A_tid + eid * T.M + kth * 8 * T.M
    // [NEXT] B_tid + eid * T.N + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32) + (T.wid * 4 * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32) + (T.wid * 4 * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 32 + T.lid < T.N) B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
    Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 256;
        Bsmem += (2 * (kth & 1) - 1) * 256;
        Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
        Bsmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 256 * 2);
    
    // [NEXT] A_tid + eid * T.M + kth * 8 * T.M
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 32) + (T.wid * 4 * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 32 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 32 + T.lid < T.M)               A_valid |= (1u << eid);
        if (T.bcid * 32 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid * 4 + eid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
        }
        if ((B_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 256;
        Bsmem += (2 * (kth & 1) - 1) * (256 + 32);
        Asmem[T.wid * 4 * 32 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 4 * 32 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 4 * 32 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 4 * 32 + T.lid + 3 * 32] = Atrans[3];
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 36 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        Creg[row][0] *= alpha;
        Creg[row][1] *= alpha;
        Creg[row][2] *= alpha;
        Creg[row][3] *= alpha;
    }
}

__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 512 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 512 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (512 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS,
    const GEMM_Order order, const uint32_t batchCount
) {
    const dim3 block_size(64, 1, 1);
    const dim3 grid_size((N + 31) / 32, (M + 31) / 32, batchCount);
    switch (order) {
    case GEMM_Order::RRR:
        sgemm_rrr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RRC:
        sgemm_rrc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RCR:
        sgemm_rcr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::RCC:
        sgemm_rcc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CRR:
        sgemm_crr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CRC:
        sgemm_crc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CCR:
        sgemm_ccr_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    case GEMM_Order::CCC:
        sgemm_ccc_kernel<<<grid_size, block_size>>>(A, B, C, alpha, M, N, K, aS, bS, cS); break;
    default: break;
    }
}

} // namespace sgemm_32x32_4x4
