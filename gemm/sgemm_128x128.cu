#pragma once

#include <cuda.h>
#include "gemm_utils.cu"
#include "../utils/buffer.cu"
#include "../utils/ptx.cu"

namespace sgemm_128x128_8x8 {

using buffer::SharedMemory;

/* [WHEN] General */
struct TileIndex {
    uint32_t brid, bcid, tid, wid, lid;
    uint32_t wrows, wcols, wrid, wcid, lrid, lcid;
    __device__ TileIndex() {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x;
        tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        wrid = wid / 4; wcid = wid % 4;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
    }
};

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[8][8], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    // 使用 32x128 共享内存搬运 128x128 数据（需 4 次），每次每线程写回 2x8 数据 Creg[r][:], Creg[r + 4][:]
    // [NEXT] C_smem_st + (tile_rid * wrows * 128 + tile_cid * wcols * 4) * sizeof(float)
    uint32_t C_smem_st = ptx::smem_addr(smem_buf + (wrid * wrows * 2 * 128 + wcid * wcols * 8) + (lrid * 128 + lcid * 4));
    float *C_block = C + (blockIdx.z * cS + brid * 128 * N + bcid * 128);
    for (uint32_t r = 0; r < 4; ++r) {
        __syncthreads();
        // 将数据写入到共享内存
        ptx::st_smem(Creg[r][0], Creg[r][1], Creg[r][2], Creg[r][3], C_smem_st + (0 * wrows * 128 + 0 * wcols * 4) * sizeof(float));
        ptx::st_smem(Creg[r][4], Creg[r][5], Creg[r][6], Creg[r][7], C_smem_st + (0 * wrows * 128 + 1 * wcols * 4) * sizeof(float));
        ptx::st_smem(Creg[r+4][0], Creg[r+4][1], Creg[r+4][2], Creg[r+4][3], C_smem_st + (1 * wrows * 128 + 0 * wcols * 4) * sizeof(float));
        ptx::st_smem(Creg[r+4][4], Creg[r+4][5], Creg[r+4][6], Creg[r+4][7], C_smem_st + (1 * wrows * 128 + 1 * wcols * 4) * sizeof(float));
        __syncthreads();
        // 使用 2x128 排列的线程搬运 32x128 共享内存（需 16 次），每次每线程写回 1 个数据
        #pragma unroll
        for (uint32_t gmem_row = r; gmem_row < 128; gmem_row += 4 * 2) {
            ptx::st_gmem(
                *reinterpret_cast<float*>(smem_buf + gmem_row / 4 * 128 + tid),
                C_block + (gmem_row + tid / 128 * 4) * N + (tid % 128),
                (brid * 128 + gmem_row + tid / 128 * 4 < M) && (bcid * 128 + tid % 128 < N)
            );
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[8][8], float *smem_buf, float *C,
    const uint32_t &M, const uint32_t &N, const uint32_t &cS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    // 使用 128x32 共享内存搬运 128x128 数据（需 4 次），每次每线程写回 8x2 数据 Creg[:][c], Creg[:][c + 4]
    // [NEXT] C_smem_st + (tile_cid * wcols * 128 + tile_rid * wrows * 4) * sizeof(float)
    uint32_t C_smem_st = ptx::smem_addr(smem_buf + (wcid * wcols * 2 * 128 + wrid * wrows * 8) + (lcid * 128 + lrid * 4));
    float *C_block = C + (blockIdx.z * cS + bcid * 128 * M + brid * 128);
    for (uint32_t c = 0; c < 4; ++c) {
        __syncthreads();
        // 将数据写入到共享内存
        ptx::st_smem(Creg[0][c], Creg[1][c], Creg[2][c], Creg[3][c], C_smem_st + (0 * wcols * 128 + 0 * wrows * 4) * sizeof(float));
        ptx::st_smem(Creg[4][c], Creg[5][c], Creg[6][c], Creg[7][c], C_smem_st + (0 * wcols * 128 + 1 * wrows * 4) * sizeof(float));
        ptx::st_smem(Creg[0][c+4], Creg[1][c+4], Creg[2][c+4], Creg[3][c+4], C_smem_st + (1 * wcols * 128 + 0 * wrows * 4) * sizeof(float));
        ptx::st_smem(Creg[4][c+4], Creg[5][c+4], Creg[6][c+4], Creg[7][c+4], C_smem_st + (1 * wcols * 128 + 1 * wrows * 4) * sizeof(float));
        __syncthreads();
        // 使用 128x2 排列的线程搬运 128x32 共享内存（需 16 次），每次每线程写回 1 个数据
        #pragma unroll
        for (uint32_t gmem_column = c; gmem_column < 128; gmem_column += 4 * 2) {
            ptx::st_gmem(
                *reinterpret_cast<float*>(smem_buf + gmem_column / 4 * 128 + tid),
                C_block + (gmem_column + tid / 128 * 4) * M + (tid % 128),
                (brid * 128 + tid % 128 < M) && (bcid * 128 + gmem_column + tid / 128 * 4 < N)
            );
        }
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[8][8], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 1024 * 4);
    // [NEXT] A_smem_st
    // [NEXT] B_smem_st + eid * 32 * sizeof(float)
    uint32_t A_smem_st = ptx::smem_addr(A_smem + tid % 8 * 132 + tid / 8 * 4);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + wid * 128 + lid);
    // [NEXT] A_smem_ld + (tile_rid * wrows * 4 + kid * ldA) * sizeof(float)
    // [NEXT] B_smem_ld + (tile_cid * wcols * 4 + kid * ldB) * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 8 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 8 + lcid * 4);
    // [NEXT] A_tid + eid * K + kth * 8
    // [NEXT] B_tid + eid * 32 + kth * 8 * N
    const float *A_tid = A + (blockIdx.z * aS + brid * 128 * K) + (tid / 8 * 4 * K + tid % 8);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 128) + (wid * N + lid);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 128 + tid / 8 * 4 + eid < M) << eid;
        B_valid |= (uint32_t)(bcid * 128 + lid + eid * 32 < N)    << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[8] = {0.f}, Breg[8] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * K, (A_valid & (1u << eid)) && (tid % 8 < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * 32, (B_valid & (1u << eid)) && (wid < kstart));
    }
    // 将预取数据写入到共享内存
    ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
    }
    __syncthreads();
    // 切换缓冲区
    A_smem_st ^= 0x2000;
    B_smem_st ^= 0x1000;
    // 数据指针向后移动 k 个数据
    A_tid += kstart;
    B_tid += kstart * N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * K, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * 32, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 128) * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 8; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 8; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
        }
        __syncthreads();
        // 切换缓冲区
        A_smem_st ^= 0x2000;
        B_smem_st ^= 0x1000;
        A_smem_ld ^= 0x2000;
        B_smem_ld ^= 0x1000;
        // 数据指针向后移动 k 个数据
        A_tid += 8;
        B_tid += 8 * N;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 128) * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 8; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 8; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 8; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 8; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[8][8], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 1024 * 4);
    // [NEXT] A_smem_st
    // [NEXT] B_smem_st
    uint32_t A_smem_st = ptx::smem_addr(A_smem + tid % 8 * 132 + tid / 8 * 4);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + tid % 8 * 132 + tid / 8 * 4);
    // [NEXT] A_smem_ld + (tile_rid * wrows * 4 + kid * ldA) * sizeof(float)
    // [NEXT] B_smem_ld + (tile_cid * wcols * 4 + kid * ldB) * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 8 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 8 + lcid * 4);
    // [NEXT] A_tid + eid * K + kth * 8
    // [NEXT] B_tid + eid * K + kth * 8
    const float *A_tid = A + (blockIdx.z * aS + brid * 128 * K) + (tid / 8 * 4 * K + tid % 8);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 128 * K) + (tid / 8 * 4 * K + tid % 8);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 128 + tid / 8 * 4 + eid < M) << eid;
        B_valid |= (uint32_t)(bcid * 128 + tid / 8 * 4 + eid < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[8] = {0.f}, Breg[8] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * K, (A_valid & (1u << eid)) && (tid % 8 < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * K, (B_valid & (1u << eid)) && (tid % 8 < kstart));
    }
    // 将预取数据写入到共享内存
    ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
    ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
    __syncthreads();
    // 切换缓冲区
    A_smem_st ^= 0x2000;
    B_smem_st ^= 0x2000;
    // 数据指针向后移动 k 个数据
    A_tid += kstart;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * K, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * K, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 132) * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 8; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 8; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        ptx::st_smem(Atrans[0], Atrans[1], Atrans[2], Atrans[3], A_smem_st);
        ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
        __syncthreads();
        // 切换缓冲区
        A_smem_st ^= 0x2000;
        B_smem_st ^= 0x2000;
        A_smem_ld ^= 0x2000;
        B_smem_ld ^= 0x2000;
        // 数据指针向后移动 k 个数据
        A_tid += 8;
        B_tid += 8;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 132) * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 8; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 8; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 8; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 8; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[8][8], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 1024 * 2);
    // [NEXT] A_smem_st + eid * 32 * sizeof(float)
    // [NEXT] B_smem_st + eid * 32 * sizeof(float)
    uint32_t A_smem_st = ptx::smem_addr(A_smem + wid * 128 + lid);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + wid * 128 + lid);
    // [NEXT] A_smem_ld + (tile_rid * wrows * 4 + kid * ldA) * sizeof(float)
    // [NEXT] B_smem_ld + (tile_cid * wcols * 4 + kid * ldB) * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 8 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 8 + lcid * 4);
    // [NEXT] A_tid + eid * 32 + kth * 8 * M
    // [NEXT] B_tid + eid * 32 + kth * 8 * N
    const float *A_tid = A + (blockIdx.z * aS + brid * 128) + (wid * M + lid);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 128) + (wid * N + lid);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 128 + lid + eid * 32 < M) << eid;
        B_valid |= (uint32_t)(bcid * 128 + lid + eid * 32 < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[8] = {0.f}, Breg[8] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * 32, (A_valid & (1u << eid)) && (wid < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * 32, (B_valid & (1u << eid)) && (wid < kstart));
    }
    // 将预取数据写入到共享内存
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
        ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
    }
    __syncthreads();
    // 切换缓冲区
    A_smem_st ^= 0x1000;
    B_smem_st ^= 0x1000;
    // 数据指针向后移动 k 个数据
    A_tid += kstart * M;
    B_tid += kstart * N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * 32, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * 32, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 128) * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 8; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 8; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
            ptx::st_smem(Btrans[eid], B_smem_st + eid * 32 * sizeof(float));
        }
        __syncthreads();
        // 切换缓冲区
        A_smem_st ^= 0x1000;
        B_smem_st ^= 0x1000;
        A_smem_ld ^= 0x1000;
        B_smem_ld ^= 0x1000;
        // 数据指针向后移动 k 个数据
        A_tid += 8 * M;
        B_tid += 8 * N;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 128) * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 8; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 8; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 8; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 8; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[8][8], float *smem_buf, const float *A, const float *B, const float &alpha,
    const uint32_t &M, const uint32_t &N, const uint32_t &K, const uint32_t &aS, const uint32_t &bS,
    const uint32_t &brid, const uint32_t &bcid, const uint32_t &tid, const uint32_t &wid, const uint32_t &lid,
    const uint32_t &wrows, const uint32_t &wcols, const uint32_t &wrid, const uint32_t &wcid,
    const uint32_t &lrid, const uint32_t &lcid
) {
    float *A_smem = reinterpret_cast<float*>(smem_buf + 1024 * 4);
    float *B_smem = reinterpret_cast<float*>(smem_buf);
    // [NEXT] A_smem_st + eid * 32 * sizeof(float)
    // [NEXT] B_smem_st
    uint32_t A_smem_st = ptx::smem_addr(A_smem + wid * 128 + lid);
    uint32_t B_smem_st = ptx::smem_addr(B_smem + tid % 8 * 132 + tid / 8 * 4);
    // [NEXT] A_smem_ld + (tile_rid * wrows * 4 + kid * ldA) * sizeof(float)
    // [NEXT] B_smem_ld + (tile_cid * wcols * 4 + kid * ldB) * sizeof(float)
    uint32_t A_smem_ld = ptx::smem_addr(A_smem + wrid * wrows * 8 + lrid * 4);
    uint32_t B_smem_ld = ptx::smem_addr(B_smem + wcid * wcols * 8 + lcid * 4);
    // [NEXT] A_tid + eid * 32 + kth * 8 * M
    // [NEXT] B_tid + eid * K + kth * 8
    const float *A_tid = A + (blockIdx.z * aS + brid * 128) + (wid * M + lid);
    const float *B_tid = B + (blockIdx.z * bS + bcid * 128 * K) + (tid / 8 * 4 * K + tid % 8);

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0u, B_valid = 0u;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_valid |= (uint32_t)(brid * 128 + lid + eid * 32 < M)    << eid;
        B_valid |= (uint32_t)(bcid * 128 + tid / 8 * 4 + eid < N) << eid;
    }
    // 数据寄存器
    float Atrans[4] = {0.f}, Btrans[4] = {0.f};
    float Areg[8] = {0.f}, Breg[8] = {0.f};

    // 预取可能不足 8 个的数据
    uint32_t kstart = K - ((K + 7) / 8 - 1) * 8;  // [1, 2, 3, ..., 8]
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::ld_gmem_zero(Atrans[eid], A_tid + eid * 32, (A_valid & (1u << eid)) && (wid < kstart));
        ptx::ld_gmem_zero(Btrans[eid], B_tid + eid * K, (B_valid & (1u << eid)) && (tid % 8 < kstart));
    }
    // 将预取数据写入到共享内存
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
    }
    ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
    __syncthreads();
    // 切换缓冲区
    A_smem_st ^= 0x1000;
    B_smem_st ^= 0x2000;
    // 数据指针向后移动 k 个数据
    A_tid += kstart * M;
    B_tid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (K + 7) / 8; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem(Atrans[eid], A_tid + eid * 32, A_valid & (1u << eid));
            ptx::ld_gmem(Btrans[eid], B_tid + eid * K, B_valid & (1u << eid));
        }
        // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
        #pragma unroll
        for (uint32_t kid = 0; kid < 8; ++kid) {
            ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 128) * sizeof(float));
            ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 132) * sizeof(float));
            ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 132) * sizeof(float));
            #pragma unroll
            for (uint32_t rid = 0; rid < 8; ++rid) {
                #pragma unroll
                for (uint32_t cid = 0; cid < 8; ++cid) {
                    Creg[rid][cid] += Areg[rid] * Breg[cid];
                }
            }
        }
        // 将预取数据写入到共享内存
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(Atrans[eid], A_smem_st + eid * 32 * sizeof(float));
        }
        ptx::st_smem(Btrans[0], Btrans[1], Btrans[2], Btrans[3], B_smem_st);
        __syncthreads();
        // 切换缓冲区
        A_smem_st ^= 0x1000;
        B_smem_st ^= 0x2000;
        A_smem_ld ^= 0x1000;
        B_smem_ld ^= 0x2000;
        // 数据指针向后移动 k 个数据
        A_tid += 8 * M;
        B_tid += 8;
    }
    // 每个线程计算 C 的子区域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 8; ++kid) {
        ptx::ld_smem(Areg[0], Areg[1], Areg[2], Areg[3], A_smem_ld + (0 * wrows * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Areg[4], Areg[5], Areg[6], Areg[7], A_smem_ld + (1 * wrows * 4 + kid * 128) * sizeof(float));
        ptx::ld_smem(Breg[0], Breg[1], Breg[2], Breg[3], B_smem_ld + (0 * wcols * 4 + kid * 132) * sizeof(float));
        ptx::ld_smem(Breg[4], Breg[5], Breg[6], Breg[7], B_smem_ld + (1 * wcols * 4 + kid * 132) * sizeof(float));
        #pragma unroll
        for (uint32_t rid = 0; rid < 8; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 8; ++cid) {
                Creg[rid][cid] += Areg[rid] * Breg[cid];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t rid = 0; rid < 8; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 8; ++cid) {
            Creg[rid][cid] *= alpha;
        }
    }
}

__launch_bounds__(256, 2)
__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_rrr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 8>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_rcr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 8>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_rcr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_crr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 4>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_crr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_ccr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rr(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__launch_bounds__(256, 2)
__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = SharedMemory<float, 1024 * 6>().pointer();
    TileIndex T;
    float Creg[8][8] = {0.f};
    compute_block_ccr(
        Creg, smem_buf, A, B, alpha, M, N, K, aS, bS,
        T.brid, T.bcid, T.tid, T.wid, T.lid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
    store_result_smem_rc(
        Creg, smem_buf, C, M, N, cS, 
        T.brid, T.bcid, T.tid, T.wrows, T.wcols, T.wrid, T.wcid, T.lrid, T.lcid
    );
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const uint32_t aS, const uint32_t bS, const uint32_t cS,
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