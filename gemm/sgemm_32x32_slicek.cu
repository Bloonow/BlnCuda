#pragma once

#include <cuda.h>
#include "gemm_utils.cu"
#include "../utils/buffer.cu"

namespace sgemm_32x32_4x8_SliceK {

/* [WHEN] 48 < K <= 256 */
struct ShapeLayoutSliceK {
    uint32_t brid, bcid, tid, wid, lid;
    uint32_t wrows, wcols, lrid, lcid;
    uint32_t M, N, K, aS, bS, cS;
    // uint32_t slice_len, slice_idx;
    __device__ ShapeLayoutSliceK(const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS) {
        // 线程块与线程的标识
        brid = blockIdx.y; bcid = blockIdx.x; tid = threadIdx.x; wid = tid / 32; lid = tid % 32;
        // 线程束的排列布局
        wrows = 8; wcols = 4;
        lrid = (lid % 16) / 2;
        lcid = (lid / 16) * 2 + (lid % 2);
        // 矩阵形状与跨步
        this->M = M; this->N = N; this->K = K;
        this->aS = aS; this->bS = bS; this->cS = cS;
        // 沿着维度轴 K 切片的标识
        // slice_len = 4; slice_idx = wid;
    }
};

__device__ __forceinline__
void reduce_over_warp(
    float *smem, const uint32_t num_warp, const uint32_t num_datum, const uint32_t wid, const uint32_t lid
) {
    /* 在一个 Block 内的所有 Warp 之上进行归约，假设 num_datum 为 4 的整数倍 */
    for (uint32_t offset = num_warp / 2; offset >= 1; offset /= 2) {
        if (wid < offset) {
            #pragma unroll
            for (uint32_t i = lid * 4; i < num_datum; i += warpSize * 4) {
                float4 my = *reinterpret_cast<float4*>(smem + wid * num_datum + i);
                float4 other = *reinterpret_cast<float4*>(smem + (wid + offset) * num_datum + i);
                my.x += other.x; my.y += other.y; my.z += other.z; my.w += other.w;
                *reinterpret_cast<float4*>(smem + wid * num_datum + i) = my;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[2][4][4], float *smem_buf, float *C,
    const uint32_t M, const uint32_t N, const uint32_t cS,
    const uint32_t brid, const uint32_t bcid, const uint32_t tid, const uint32_t wid, const uint32_t lid,
    const uint32_t wcols, const uint32_t lrid, const uint32_t lcid
) {
    // 存在 slice_num 份矩阵 C 子区域的部分结果，需先使用共享内存对其进行归约   
    float *Csmem = smem_buf + 1024 * wid;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + brid * 32 * N + bcid * 32);

    float4 trans1, trans2;
    __syncthreads();
    // 首先，所有线程先将部分结果数据写入到共享内存，每个线程负责写回 Creg[2][4][4] 的数据
    #pragma unroll
    for (uint32_t row = 0; row < 4; ++row) {
        trans1.x = Creg[0][row][0]; trans1.y = Creg[0][row][1]; trans1.z = Creg[0][row][2]; trans1.w = Creg[0][row][3];
        trans2.x = Creg[1][row][0]; trans2.y = Creg[1][row][1]; trans2.z = Creg[1][row][2]; trans2.w = Creg[1][row][3];
        *reinterpret_cast<float4*>(Csmem + (0 * wcols * 4 + lcid * 4) + (lrid * 4 * 32 + row * 32)) = trans1;
        *reinterpret_cast<float4*>(Csmem + (1 * wcols * 4 + lcid * 4) + (lrid * 4 * 32 + row * 32)) = trans2;
    }
    __syncthreads();
    // 在 slice_num 个线程束之上进行归约，函数结束时存在显式 __syncthreads() 同步
    reduce_over_warp(smem_buf, 4, 1024, wid, lid);
    // 将数据从共享内存转移到全局内存
    // 使用 4x32 排列的线程搬运 32x32 共享内存，共需 8 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_row = 0; gmem_row < 32; gmem_row += 4) {
        if ((brid * 32 + gmem_row + tid / 32 < M) && (bcid * 32 + tid % 32 < N)) {
            *(C_block + (gmem_row + tid / 32) * N + (tid % 32)) = *(smem_buf + gmem_row * 32 + tid);
        }
    }
}

__device__ __forceinline__
void store_result_smem_rc(
    float Creg[2][4][4], float *smem_buf, float *C,
    const uint32_t M, const uint32_t N, const uint32_t cS,
    const uint32_t brid, const uint32_t bcid, const uint32_t tid, const uint32_t wid, const uint32_t lid,
    const uint32_t wcols, const uint32_t lrid, const uint32_t lcid
) {
    // 存在 slice_num 份矩阵 C 子区域的部分结果，需先使用共享内存对其进行归约   
    float *Csmem = smem_buf + 1024 * wid;
    // 写回矩阵 C 的子区域，使用 32x32 共享内存搬运 32x32 数据，共需 1 次
    float *C_block = C + (blockIdx.z * cS + bcid * 32 * M + brid * 32);

    float4 trans1, trans2;
    __syncthreads();
    // 首先，所有线程先将部分结果数据写入到共享内存，每个线程负责写回 Creg[2][4][4] 的数据
    #pragma unroll
    for (uint32_t column = 0; column < 4; ++column) {
        trans1.x = Creg[0][0][column]; trans1.y = Creg[0][1][column]; trans1.z = Creg[0][2][column]; trans1.w = Creg[0][3][column];
        trans2.x = Creg[1][0][column]; trans2.y = Creg[1][1][column]; trans2.z = Creg[1][2][column]; trans2.w = Creg[1][3][column];
        *reinterpret_cast<float4*>(Csmem + (0 * wcols * 4 * 32 + lcid * 4 * 32 + column * 32) + (lrid * 4)) = trans1;
        *reinterpret_cast<float4*>(Csmem + (1 * wcols * 4 * 32 + lcid * 4 * 32 + column * 32) + (lrid * 4)) = trans2;
    }
    __syncthreads();
    // 在 slice_num 个线程束之上进行归约，函数结束时存在显式 __syncthreads() 同步
    reduce_over_warp(smem_buf, 4, 1024, wid, lid);
    // 将数据从共享内存转移到全局内存
    // 使用 32x4 排列的线程搬运 32x32 共享内存，共需 8 次，每次每个线程写回 1 个数据
    #pragma unroll
    for (uint32_t gmem_column = 0; gmem_column < 32; gmem_column += 4) {

        if ((brid * 32 + tid % 32 < M) && (bcid * 32 + gmem_column + tid / 32 < N)) {
            *(C_block + (gmem_column + tid / 32) * M + (tid % 32)) = *(smem_buf + gmem_column * 32 + tid);
        }
    }
}

__device__ __forceinline__
void compute_tile_crr(
    float Creg[2][4][4], float *Asmem, float *Bsmem, const uint32_t ldA, const uint32_t ldB,
    const uint32_t wcols, const uint32_t lrid, const uint32_t lcid
) {
    float4 Areg, Breg[2];
    // 每个线程计算 C 的子域，采用向量外积方式，在 K_block 维度上循环迭代
    #pragma unroll
    for (uint32_t kid = 0; kid < 4; ++kid) {
        Areg = *reinterpret_cast<float4*>(Asmem + lrid * 4 + kid * ldA);
        Breg[0] = *reinterpret_cast<float4*>(Bsmem + 0 * wcols * 4 + lcid * 4 + kid * ldB);
        Breg[1] = *reinterpret_cast<float4*>(Bsmem + 1 * wcols * 4 + lcid * 4 + kid * ldB);
        #pragma unroll
        for (uint32_t cpj = 0; cpj < 2; ++cpj) {
            Creg[cpj][0][0] += Areg.x * Breg[cpj].x;
            Creg[cpj][0][1] += Areg.x * Breg[cpj].y;
            Creg[cpj][0][2] += Areg.x * Breg[cpj].z;
            Creg[cpj][0][3] += Areg.x * Breg[cpj].w;
            Creg[cpj][1][0] += Areg.y * Breg[cpj].x;
            Creg[cpj][1][1] += Areg.y * Breg[cpj].y;
            Creg[cpj][1][2] += Areg.y * Breg[cpj].z;
            Creg[cpj][1][3] += Areg.y * Breg[cpj].w;
            Creg[cpj][2][0] += Areg.z * Breg[cpj].x;
            Creg[cpj][2][1] += Areg.z * Breg[cpj].y;
            Creg[cpj][2][2] += Areg.z * Breg[cpj].z;
            Creg[cpj][2][3] += Areg.z * Breg[cpj].w;
            Creg[cpj][3][0] += Areg.w * Breg[cpj].x;
            Creg[cpj][3][1] += Areg.w * Breg[cpj].y;
            Creg[cpj][3][2] += Areg.w * Breg[cpj].z;
            Creg[cpj][3][3] += Areg.w * Breg[cpj].w;
        }
    }
}

__device__ __forceinline__
void compute_block_rrr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float alpha, const ShapeLayoutSliceK &SL
) {
    float *Asmem = smem_buf + 1024 * SL.wid;
    float *Bsmem = smem_buf + 1024 * SL.wid + (128 + 32) * 2;

    // [NEXT] A_lid + eid * SL.K + kth * 16       + slice_idx * 4
    // [NEXT] B_lid + eid * SL.N + kth * 16 * SL.N + slice_idx * 4 * SL.N
    const float *A_lid = A + (blockIdx.z * SL.aS + SL.brid * 32 * SL.K) + (SL.lid / 4 * 4 * SL.K + SL.lid % 4);
    const float *B_lid = B + (blockIdx.z * SL.bS + SL.bcid * 32) + SL.lid;
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (SL.brid * 32 + SL.lid / 4 * 4 + eid < SL.M) A_valid |= (1u << eid);
        if (SL.bcid * 32 + SL.lid < SL.N)               B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    uint32_t kstart = SL.K - ((SL.K + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (SL.wid * 4 + SL.lid % 4 < kstart)) {
            Atrans[eid] = *(A_lid + SL.wid * 4 + eid * SL.K);
        }
        if ((B_valid & (1u << eid)) && (SL.wid * 4 + eid < kstart)) {
            Btrans[eid] = *(B_lid + SL.wid * 4 * SL.N + eid * SL.N);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
    Bsmem[SL.lid + 0 * 32] = Btrans[0];
    Bsmem[SL.lid + 1 * 32] = Btrans[1];
    Bsmem[SL.lid + 2 * 32] = Btrans[2];
    Bsmem[SL.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_lid += kstart;
    B_lid += kstart * SL.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (SL.K + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *(A_lid + SL.wid * 4 + eid * SL.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *(B_lid + SL.wid * 4 * SL.N + eid * SL.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, SL.wcols, SL.lrid, SL.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (128 + 32);
        Bsmem += (2 * (kth & 1) - 1) * 128;
        *reinterpret_cast<float4*>(Asmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
        Bsmem[SL.lid + 0 * 32] = Btrans[0];
        Bsmem[SL.lid + 1 * 32] = Btrans[1];
        Bsmem[SL.lid + 2 * 32] = Btrans[2];
        Bsmem[SL.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_lid += 16;
        B_lid += 16 * SL.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 32, SL.wcols, SL.lrid, SL.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (uint32_t row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_rcr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float alpha, const ShapeLayoutSliceK &SL
) {
    float *Asmem = smem_buf + 1024 * SL.wid;
    float *Bsmem = smem_buf + 1024 * SL.wid + (128 + 32) * 2;

    // [NEXT] A_lid + eid * SL.K + kth * 16 + slice_idx * 4
    // [NEXT] B_lid + eid * SL.K + kth * 16 + slice_idx * 4
    const float *A_lid = A + (blockIdx.z * SL.aS + SL.brid * 32 * SL.K) + (SL.lid / 4 * 4 * SL.K + SL.lid % 4);
    const float *B_lid = B + (blockIdx.z * SL.bS + SL.bcid * 32 * SL.K) + (SL.lid / 4 * 4 * SL.K + SL.lid % 4);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (SL.brid * 32 + SL.lid / 4 * 4 + eid < SL.M) A_valid |= (1u << eid);
        if (SL.bcid * 32 + SL.lid / 4 * 4 + eid < SL.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    uint32_t kstart = SL.K - ((SL.K + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (SL.wid * 4 + SL.lid % 4 < kstart)) {
            Atrans[eid] = *(A_lid + SL.wid * 4 + eid * SL.K);
        }
        if ((B_valid & (1u << eid)) && (SL.wid * 4 + SL.lid % 4 < kstart)) {
            Btrans[eid] = *(B_lid + SL.wid * 4 + eid * SL.K);
        }
    }

    // 将预取数据写入到共享内存
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Asmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
    *reinterpret_cast<float4*>(Bsmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_lid += kstart;
    B_lid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (SL.K + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *(A_lid + SL.wid * 4 + eid * SL.K);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *(B_lid + SL.wid * 4 + eid * SL.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, SL.wcols, SL.lrid, SL.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * (128 + 32);
        Bsmem += (2 * (kth & 1) - 1) * (128 + 32);
        *reinterpret_cast<float4*>(Asmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Atrans);
        *reinterpret_cast<float4*>(Bsmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_lid += 16;
        B_lid += 16;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 36, 36, SL.wcols, SL.lrid, SL.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (uint32_t row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_crr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float alpha, const ShapeLayoutSliceK &SL
) {
    float *Asmem = smem_buf + 1024 * SL.wid;
    float *Bsmem = smem_buf + 1024 * SL.wid + 128 * 2;

    // [NEXT] A_lid + eid * SL.M + kth * 16 * SL.M + slice_idx * 4 * SL.M
    // [NEXT] B_lid + eid * SL.N + kth * 16 * SL.N + slice_idx * 4 * SL.N
    const float *A_lid = A + (blockIdx.z * SL.aS + SL.brid * 32) + SL.lid;
    const float *B_lid = B + (blockIdx.z * SL.bS + SL.bcid * 32) + SL.lid;
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (SL.brid * 32 + SL.lid < SL.M) A_valid |= (1u << eid);
        if (SL.bcid * 32 + SL.lid < SL.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    uint32_t kstart = SL.K - ((SL.K + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (SL.wid * 4 + eid < kstart)) {
            Atrans[eid] = *(A_lid + SL.wid * 4 * SL.M + eid * SL.M);
        }
        if ((B_valid & (1u << eid)) && (SL.wid * 4 + eid < kstart)) {
            Btrans[eid] = *(B_lid + SL.wid * 4 * SL.N + eid * SL.N);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[SL.lid + 0 * 32] = Atrans[0];
    Asmem[SL.lid + 1 * 32] = Atrans[1];
    Asmem[SL.lid + 2 * 32] = Atrans[2];
    Asmem[SL.lid + 3 * 32] = Atrans[3];
    Bsmem[SL.lid + 0 * 32] = Btrans[0];
    Bsmem[SL.lid + 1 * 32] = Btrans[1];
    Bsmem[SL.lid + 2 * 32] = Btrans[2];
    Bsmem[SL.lid + 3 * 32] = Btrans[3];
    __syncthreads();
    A_lid += kstart * SL.M;
    B_lid += kstart * SL.N;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (SL.K + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *(A_lid + SL.wid * 4 * SL.M + eid * SL.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *(B_lid + SL.wid * 4 * SL.N + eid * SL.N);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, SL.wcols, SL.lrid, SL.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 128;
        Bsmem += (2 * (kth & 1) - 1) * 128;
        Asmem[SL.lid + 0 * 32] = Atrans[0];
        Asmem[SL.lid + 1 * 32] = Atrans[1];
        Asmem[SL.lid + 2 * 32] = Atrans[2];
        Asmem[SL.lid + 3 * 32] = Atrans[3];
        Bsmem[SL.lid + 0 * 32] = Btrans[0];
        Bsmem[SL.lid + 1 * 32] = Btrans[1];
        Bsmem[SL.lid + 2 * 32] = Btrans[2];
        Bsmem[SL.lid + 3 * 32] = Btrans[3];
        __syncthreads();
        A_lid += 16 * SL.M;
        B_lid += 16 * SL.N;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 32, SL.wcols, SL.lrid, SL.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (uint32_t row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__device__ __forceinline__
void compute_block_ccr(
    float Creg[2][4][4], float *smem_buf, const float *A, const float *B, const float alpha, const ShapeLayoutSliceK &SL
) {
    float *Asmem = smem_buf + 1024 * SL.wid;
    float *Bsmem = smem_buf + 1024 * SL.wid + 128 * 2;

    // [NEXT] A_lid + eid * SL.M + kth * 16 * SL.M + slice_idx * 4 * SL.M
    // [NEXT] B_lid + eid * SL.K + kth * 16 + slice_idx * 4
    const float *A_lid = A + (blockIdx.z * SL.aS + SL.brid * 32) + SL.lid;
    const float *B_lid = B + (blockIdx.z * SL.bS + SL.bcid * 32 * SL.K) + (SL.lid / 4 * 4 * SL.K + SL.lid % 4);
    float Atrans[4] = {}, Btrans[4] = {};

    // valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据
    uint32_t A_valid = 0U, B_valid = 0U;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if (SL.brid * 32 + SL.lid < SL.M)               A_valid |= (1u << eid);
        if (SL.bcid * 32 + SL.lid / 4 * 4 + eid < SL.N) B_valid |= (1u << eid);
    }

    // 一次完整的 slice_num = 4 迭代在 K 的维度上读取 slice_num * slice_len = 4 * 4 = 16 的数据，首先处理刚开始的可能情况
    uint32_t kstart = SL.K - ((SL.K + 15) / 16 - 1) * 16;  // [1, 2, 3, ..., 16]
    // 预取可能不足 16 个的元素
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        if ((A_valid & (1u << eid)) && (SL.wid * 4 + eid < kstart)) {
            Atrans[eid] = *(A_lid + SL.wid * 4 * SL.M + eid * SL.M);
        }
        if ((B_valid & (1u << eid)) && (SL.wid * 4 + SL.lid % 4 < kstart)) {
            Btrans[eid] = *(B_lid + SL.wid * 4 + eid * SL.K);
        }
    }

    // 将预取数据写入到共享内存
    Asmem[SL.lid + 0 * 32] = Atrans[0];
    Asmem[SL.lid + 1 * 32] = Atrans[1];
    Asmem[SL.lid + 2 * 32] = Atrans[2];
    Asmem[SL.lid + 3 * 32] = Atrans[3];
    // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
    *reinterpret_cast<float4*>(Bsmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
    __syncthreads();
    A_lid += kstart * SL.M;
    B_lid += kstart;

    // 在 K 的维度轴上进行循环迭代，计算矩阵 C 的子区域
    for (uint32_t kth = 1; kth < (SL.K + 15) / 16; ++kth) {
        // 预取 kth 的数据
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            if (A_valid & (1u << eid)) {
                Atrans[eid] = *(A_lid + SL.wid * 4 * SL.M + eid * SL.M);
            }
            if (B_valid & (1u << eid)) {
                Btrans[eid] = *(B_lid + SL.wid * 4 + eid * SL.K);
            }
        }
        // 计算 C 的子区域
        compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, SL.wcols, SL.lrid, SL.lcid);
        // 将预取数据写入到共享内存
        Asmem += (2 * (kth & 1) - 1) * 128;
        Bsmem += (2 * (kth & 1) - 1) * (128 + 32);
        Asmem[SL.lid + 0 * 32] = Atrans[0];
        Asmem[SL.lid + 1 * 32] = Atrans[1];
        Asmem[SL.lid + 2 * 32] = Atrans[2];
        Asmem[SL.lid + 3 * 32] = Atrans[3];
        // 此处采用 32 + 4 是因为使用 4 做偏移时，保证可使用 float4 向量化读写共享内存，且使用 float4 写入时不存在 bank 冲突
        *reinterpret_cast<float4*>(Bsmem + SL.lid % 4 * 36 + SL.lid / 4 * 4) = *reinterpret_cast<float4*>(Btrans);
        __syncthreads();
        A_lid += 16 * SL.M;
        B_lid += 16;
    }
    // 计算 C 的子区域
    compute_tile_crr(Creg, Asmem, Bsmem, 32, 36, SL.wcols, SL.lrid, SL.lcid);

    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t cpj = 0; cpj < 2; ++cpj) {
        #pragma unroll
        for (uint32_t row = 0; row < 4; ++row) {
            Creg[cpj][row][0] *= alpha;
            Creg[cpj][row][1] *= alpha;
            Creg[cpj][row][2] *= alpha;
            Creg[cpj][row][3] *= alpha;
        }
    }
}

__global__ void sgemm_rrr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rr(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_rrc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_rrr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rc(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_rcr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rr(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_rcc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_rcr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rc(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_crr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rr(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_crc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_crr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rc(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_ccr_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rr(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__global__ void sgemm_ccc_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS
) {
    float *smem_buf = buffer::SharedMemory<float, 1024 * 4>().pointer();
    ShapeLayoutSliceK SL(M, N, K, aS, bS, cS);
    float Creg[2][4][4] = {};
    compute_block_ccr(Creg, smem_buf, A, B, alpha, SL);
    store_result_smem_rc(Creg, smem_buf, C, SL.M, SL.N, SL.cS, SL.brid, SL.bcid, SL.tid, SL.wid, SL.lid, SL.wcols, SL.lrid, SL.lcid);
}

__host__ void sgemm_cuda(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t aS, const uint32_t bS, const uint32_t cS,
    const GEMM_Order order, const uint32_t batchCount
) {
    const dim3 block_size(128, 1, 1);
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

} // namespace sgemm_32x32_4x8_SliceK
