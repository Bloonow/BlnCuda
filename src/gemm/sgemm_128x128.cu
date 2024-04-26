#pragma once

#include <cuda.h>
#include "gemm_utils.cu"
#include "../utils/buffer.cu"

namespace sgemm_128x128_8x8 {

using buffer::SharedMemory;

/* [WHEN] General */
struct TileIndex {
    uint32_t brid, bcid, tid, wid, lid;
    uint32_t wrows, wcols, wrid, wcid, lrid, lcid;
    uint32_t M, N, K, aS, bS, cS;
    __device__ TileIndex(const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS) {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x; tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 4; wcid = wid % 4;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
    }
};

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[2][2][4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 trans1, trans2;
    // 写回矩阵 C 的子区域，使用 32x128 共享内存搬运 128x128 数据，共需 4 次，每次每个线程写回 2x8 数据 Creg[0:1][0:1][row][0:3]
    // 对应 [cpi, cpj] = [0:1, 0:1] 时为 1x4 形状的数据
    float *C_block = C + (blockIdx.z * cS + brid * 128 * N + bcid * 128);
    for (uint32_t row = 0; row < 4; ++row) {
        __syncthreads();
        // 将数据写入到共享内存，存在bank冲突，待改进
        #pragma unroll
        for (uint32_t cpi = 0; cpi < 2; ++cpi) {
            trans1.x = Creg[cpi][0][row][0]; trans1.y = Creg[cpi][0][row][1]; trans1.z = Creg[cpi][0][row][2]; trans1.w = Creg[cpi][0][row][3];
            trans2.x = Creg[cpi][1][row][0]; trans2.y = Creg[cpi][1][row][1]; trans2.z = Creg[cpi][1][row][2]; trans2.w = Creg[cpi][1][row][3];
            *reinterpret_cast<float4*>(
                smem_buf + (wrid * wrows * 2 * 128 + wcid * wcols * 8) + (cpi * wrows * 128 + 0 * wcols * 4) + (lrid * 128 + lcid * 4)
            ) = trans1;
            *reinterpret_cast<float4*>(
                smem_buf + (wrid * wrows * 2 * 128 + wcid * wcols * 8) + (cpi * wrows * 128 + 1 * wcols * 4) + (lrid * 128 + lcid * 4)
            ) = trans2;
        }
        __syncthreads();
        // 将数据从共享内存转移到全局内存
        // 使用 2x128 排列的线程搬运 32x128 共享内存，共需 16 次，每次每个线程写回 1 个数据
        #pragma unroll
        for (uint32_t gmem_row = row; gmem_row < 128; gmem_row += 4 * 2) {
            if ((brid * 128 + gmem_row + tid / 128 * 4 < M) && (bcid * 128 + tid % 128 < N)) {
                *reinterpret_cast<float*>(
                    C_block + (gmem_row + tid / 128 * 4) * N + (tid % 128)
                ) = *reinterpret_cast<float*>(smem_buf + gmem_row / 4 * 128 + tid);
            }
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[2][2][4][4], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 trans1, trans2;
    // 写回矩阵 C 的子区域，使用 128x32 共享内存搬运 128x128 数据，共需 4 次，每次每个线程写回 8x2 数据 Creg[0:1][0:1][0:3][column]
    // 对应 [cpi, cpj] = [0:1, 0:1] 时为 4x1 形状的数据
    float *C_block = C + (blockIdx.z * cS + bcid * 128 * M + brid * 128);
    for (uint32_t column = 0; column < 4; ++column) {
        __syncthreads();
        // 将数据写入到共享内存，存在bank冲突，待改进
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            trans1.x = Creg[0][cpj][0][column]; trans1.y = Creg[0][cpj][1][column]; trans1.z = Creg[0][cpj][2][column]; trans1.w = Creg[0][cpj][3][column];
            trans2.x = Creg[1][cpj][0][column]; trans2.y = Creg[1][cpj][1][column]; trans2.z = Creg[1][cpj][2][column]; trans2.w = Creg[1][cpj][3][column];
            *reinterpret_cast<float4*>(
                smem_buf + (wcid * wcols * 2 * 128 + wrid * wrows * 8) + (cpj * wcols * 128 + 0 * wrows * 4) + (lcid * 128 + lrid * 4)
            ) = trans1;
            *reinterpret_cast<float4*>(
                smem_buf + (wcid * wcols * 2 * 128 + wrid * wrows * 8) + (cpj * wcols * 128 + 1 * wrows * 4) + (lcid * 128 + lrid * 4)
            ) = trans2;
        }
        __syncthreads();
        // 将数据从共享内存转移到全局内存
        // 使用 128x2 排列的线程搬运 128x32 共享内存，共需 16 次，每次每个线程写回 1 个数据
        #pragma unroll
        for (uint32_t gmem_column = column; gmem_column < 128; gmem_column += 8) {
            if ((brid * 128 + tid % 128 < M) && (bcid * 128 + gmem_column + tid / 128 * 4 < N)) {
                *reinterpret_cast<float*>(
                    C_block + (gmem_column + tid / 128 * 4) * M + (tid % 128)
                ) = *reinterpret_cast<float*>(smem_buf + gmem_column / 4 * 128 + tid);
            }
        }
    }
}

__device__ __forceinline__
void compute_tile_overlap_crr(
    float Creg[2][2][4][4], float *Asmem, float *Bsmem, const uint32_t &ldA, const uint32_t &ldB,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    // 将共享内存中 kid = 0 数据加载到寄存器
    float4 Areg[2][2], Breg[2][2];
    Areg[0][0] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 0 * wrows * 4 + lrid * 4);
    Areg[0][1] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 1 * wrows * 4 + lrid * 4);
    Breg[0][0] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 0 * wcols * 4 + lcid * 4);
    Breg[0][1] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 1 * wcols * 4 + lcid * 4);
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        if (kid < 7) {
            Areg[(kid+1) & 1][0] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 0 * wrows * 4 + lrid * 4 + (kid+1) * ldA);
            Areg[(kid+1) & 1][1] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 1 * wrows * 4 + lrid * 4 + (kid+1) * ldA);
            Breg[(kid+1) & 1][0] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 0 * wcols * 4 + lcid * 4 + (kid+1) * ldB);
            Breg[(kid+1) & 1][1] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 1 * wcols * 4 + lcid * 4 + (kid+1) * ldB);
        }
        #pragma unroll
        for (uint32_t cpi = 0; cpi < 2; ++cpi) {
            #pragma unroll
            for (uint32_t cpj = 0; cpj < 2; ++cpj) {
                Creg[cpi][cpj][0][0] += Areg[kid & 1][cpi].x * Breg[kid & 1][cpj].x;
                Creg[cpi][cpj][0][1] += Areg[kid & 1][cpi].x * Breg[kid & 1][cpj].y;
                Creg[cpi][cpj][0][2] += Areg[kid & 1][cpi].x * Breg[kid & 1][cpj].z;
                Creg[cpi][cpj][0][3] += Areg[kid & 1][cpi].x * Breg[kid & 1][cpj].w;
                Creg[cpi][cpj][1][0] += Areg[kid & 1][cpi].y * Breg[kid & 1][cpj].x;
                Creg[cpi][cpj][1][1] += Areg[kid & 1][cpi].y * Breg[kid & 1][cpj].y;
                Creg[cpi][cpj][1][2] += Areg[kid & 1][cpi].y * Breg[kid & 1][cpj].z;
                Creg[cpi][cpj][1][3] += Areg[kid & 1][cpi].y * Breg[kid & 1][cpj].w;
                Creg[cpi][cpj][2][0] += Areg[kid & 1][cpi].z * Breg[kid & 1][cpj].x;
                Creg[cpi][cpj][2][1] += Areg[kid & 1][cpi].z * Breg[kid & 1][cpj].y;
                Creg[cpi][cpj][2][2] += Areg[kid & 1][cpi].z * Breg[kid & 1][cpj].z;
                Creg[cpi][cpj][2][3] += Areg[kid & 1][cpi].z * Breg[kid & 1][cpj].w;
                Creg[cpi][cpj][3][0] += Areg[kid & 1][cpi].w * Breg[kid & 1][cpj].x;
                Creg[cpi][cpj][3][1] += Areg[kid & 1][cpi].w * Breg[kid & 1][cpj].y;
                Creg[cpi][cpj][3][2] += Areg[kid & 1][cpi].w * Breg[kid & 1][cpj].z;
                Creg[cpi][cpj][3][3] += Areg[kid & 1][cpi].w * Breg[kid & 1][cpj].w;
            }
        }
    }
}

__device__ __forceinline__
void compute_tile_crr(
    float Creg[2][2][4][4], float *Asmem, float *Bsmem, const uint32_t &ldA, const uint32_t &ldB,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid, const uint32_t &lrid, const uint32_t &lcid
) {
    float4 Areg[2], Breg[2];
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        Areg[0] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 0 * wrows * 4 + lrid * 4 + kid * ldA);
        Areg[1] = *reinterpret_cast<float4*>(Asmem + wrid * wrows * 8 + 1 * wrows * 4 + lrid * 4 + kid * ldA);
        Breg[0] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 0 * wcols * 4 + lcid * 4 + kid * ldB);
        Breg[1] = *reinterpret_cast<float4*>(Bsmem + wcid * wcols * 8 + 1 * wcols * 4 + lcid * 4 + kid * ldB);
        #pragma unroll
        for (uint32_t cpi = 0; cpi < 2; ++cpi) {
            #pragma unroll
            for (uint32_t cpj = 0; cpj < 2; ++cpj) {
                Creg[cpi][cpj][0][0] += Areg[cpi].x * Breg[cpj].x;
                Creg[cpi][cpj][0][1] += Areg[cpi].x * Breg[cpj].y;
                Creg[cpi][cpj][0][2] += Areg[cpi].x * Breg[cpj].z;
                Creg[cpi][cpj][0][3] += Areg[cpi].x * Breg[cpj].w;
                Creg[cpi][cpj][1][0] += Areg[cpi].y * Breg[cpj].x;
                Creg[cpi][cpj][1][1] += Areg[cpi].y * Breg[cpj].y;
                Creg[cpi][cpj][1][2] += Areg[cpi].y * Breg[cpj].z;
                Creg[cpi][cpj][1][3] += Areg[cpi].y * Breg[cpj].w;
                Creg[cpi][cpj][2][0] += Areg[cpi].z * Breg[cpj].x;
                Creg[cpi][cpj][2][1] += Areg[cpi].z * Breg[cpj].y;
                Creg[cpi][cpj][2][2] += Areg[cpi].z * Breg[cpj].z;
                Creg[cpi][cpj][2][3] += Areg[cpi].z * Breg[cpj].w;
                Creg[cpi][cpj][3][0] += Areg[cpi].w * Breg[cpj].x;
                Creg[cpi][cpj][3][1] += Areg[cpi].w * Breg[cpj].y;
                Creg[cpi][cpj][3][2] += Areg[cpi].w * Breg[cpj].z;
                Creg[cpi][cpj][3][3] += Areg[cpi].w * Breg[cpj].w;
            }
        }
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[2][2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (1024 + 32) * 2);
    
    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * 32  + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 128 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 128) + (T.wid * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 128 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 128 + T.lid + eid * 32 < T.N)    B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * T.K);
        }
        if ((B_valid & (1u << eid)) && (T.wid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 128 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[T.wid * 128 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 128 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 128 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 128 + T.lid + 3 * 32] = Btrans[3];
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
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 132, 128, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (1024 + 32);
        Bsmem += (2 * (kth & 1) - 1) * 1024;
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[T.wid * 128 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 128 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 128 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 128 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 132, 128, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            #pragma unroll
            for (uint32_t row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] *= alpha;
                Creg[cpi][cpj][row][1] *= alpha;
                Creg[cpi][cpj][row][2] *= alpha;
                Creg[cpi][cpj][row][3] *= alpha;
            }
        }
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[2][2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + (1024 + 32) * 2);

    // [NEXT] A_tid + eid * T.K + kth * 8
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 128 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 128 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 128 + T.tid / 8 * 4 + eid < T.M) A_valid |= (1u << eid);
        if (T.bcid * 128 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
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
    // 此处采用 128 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
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
        compute_tile_crr(Creg, Asmem, Bsmem, 132, 132, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (1024 + 32);
        Bsmem += (2 * (kth & 1) - 1) * (1024 + 32);
        *reinterpret_cast<float4*>(Asmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Atrans);
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 132, 132, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            #pragma unroll
            for (uint32_t row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] *= alpha;
                Creg[cpi][cpj][row][1] *= alpha;
                Creg[cpi][cpj][row][2] *= alpha;
                Creg[cpi][cpj][row][3] *= alpha;
            }
        }
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[2][2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * 2);

    // [NEXT] A_tid + eid * 32 + kth * 8 * T.M
    // [NEXT] B_tid + eid * 32 + kth * 8 * T.N
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 128) + (T.wid * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 128) + (T.wid * T.N + T.lid);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 128 + T.lid + eid * 32 < T.M) A_valid |= (1u << eid);
        if (T.bcid * 128 + T.lid + eid * 32 < T.N) B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * 32);
        }
        if ((B_valid & (1u << eid)) && (T.wid < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[T.wid * 128 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 128 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 128 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 128 + T.lid + 3 * 32] = Atrans[3];
    Bsmem[T.wid * 128 + T.lid + 0 * 32] = Btrans[0];
    Bsmem[T.wid * 128 + T.lid + 1 * 32] = Btrans[1];
    Bsmem[T.wid * 128 + T.lid + 2 * 32] = Btrans[2];
    Bsmem[T.wid * 128 + T.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart * T.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * 32);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * 32);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 128, 128, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 1024;
        Bsmem += (2 * (kth & 1) - 1) * 1024;
        Asmem[T.wid * 128 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 128 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 128 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 128 + T.lid + 3 * 32] = Atrans[3];
        Bsmem[T.wid * 128 + T.lid + 0 * 32] = Btrans[0];
        Bsmem[T.wid * 128 + T.lid + 1 * 32] = Btrans[1];
        Bsmem[T.wid * 128 + T.lid + 2 * 32] = Btrans[2];
        Bsmem[T.wid * 128 + T.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8 * T.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 128, 128, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            #pragma unroll
            for (uint32_t row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] *= alpha;
                Creg[cpi][cpj][row][1] *= alpha;
                Creg[cpi][cpj][row][2] *= alpha;
                Creg[cpi][cpj][row][3] *= alpha;
            }
        }
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[2][2][4][4], float *smem_buf, const float *A, const float *B, const float &alpha, const TileIndex &T
) {
    float *Asmem = reinterpret_cast<float*>(smem_buf);
    float *Bsmem = reinterpret_cast<float*>(smem_buf + 1024 * 2);

    // [NEXT] A_tid + eid * 32 + kth * 8 * T.M
    // [NEXT] B_tid + eid * T.K + kth * 8
    const float *A_tid = A + (blockIdx.z * T.aS + T.brid * 128) + (T.wid * T.M + T.lid);
    const float *B_tid = B + (blockIdx.z * T.bS + T.bcid * 128 * T.K) + (T.tid / 8 * 4 * T.K + T.tid % 8);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (T.brid * 128 + T.lid + eid * 32 < T.M)    A_valid |= (1u << eid);
        if (T.bcid * 128 + T.tid / 8 * 4 + eid < T.N) B_valid |= (1u << eid);
    }

    uint32_t kstart = T.K - ((T.K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    // 预取可能不足 8 个的数据
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (T.wid < kstart)) {
            Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * 32);
        }
        if ((B_valid & (1u << eid)) && (T.tid % 8 < kstart)) {
            Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 128 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    Asmem[T.wid * 128 + T.lid + 0 * 32] = Atrans[0];
    Asmem[T.wid * 128 + T.lid + 1 * 32] = Atrans[1];
    Asmem[T.wid * 128 + T.lid + 2 * 32] = Atrans[2];
    Asmem[T.wid * 128 + T.lid + 3 * 32] = Atrans[3];
    *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_tid += kstart * T.M;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (T.K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *reinterpret_cast<const float*>(A_tid + eid * 32);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *reinterpret_cast<const float*>(B_tid + eid * T.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 128, 132, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 1024;
        Bsmem += (2 * (kth & 1) - 1) * (1024 + 32);
        Asmem[T.wid * 128 + T.lid + 0 * 32] = Atrans[0];
        Asmem[T.wid * 128 + T.lid + 1 * 32] = Atrans[1];
        Asmem[T.wid * 128 + T.lid + 2 * 32] = Atrans[2];
        Asmem[T.wid * 128 + T.lid + 3 * 32] = Atrans[3];
        *reinterpret_cast<float4*>(Bsmem + T.tid % 8 * 132 + T.tid / 8 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_tid += 8 * T.M;
        B_tid += 8;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 128, 132, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpi = 0; cpi < 2; ++cpi) {
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            #pragma unroll
            for (uint32_t row = 0; row < 4; ++row) {
                Creg[cpi][cpj][row][0] *= alpha;
                Creg[cpi][cpj][row][1] *= alpha;
                Creg[cpi][cpj][row][2] *= alpha;
                Creg[cpi][cpj][row][3] *= alpha;
            }
        }
    }
}

__launch_bounds__(256, 2)
__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 64) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 2048 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 2048 * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rr(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__launch_bounds__(256, 2)
__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, (2048 + 32) * 2>().pointer();
    TileIndex T(M, N, K, aS, bS, cS);
    float Creg[2][2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, T);
    store_result_smem_rc(Creg, smem_buf, C, T.M, T.N, T.cS, T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid);
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS,
    const GEMM_Order order, const uint32_t batchCount
) {
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 127) / 128, (M + 127) / 128, batchCount);
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

} // namespace sgemm_128x128_8x8