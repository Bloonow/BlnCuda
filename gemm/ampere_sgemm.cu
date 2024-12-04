#pragma once

#include <cuda.h>
#include "../utils/buffer.cu"
#include "../utils/ptx.cu"

namespace sgemm_128x256_16x8 {

struct StgFrag {
    float data[4][4];
    __device__ __forceinline__
    StgFrag(const float (&C_frag)[16][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m, uint32_t n, uint32_t m_idx, uint32_t n_idx) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        ptx::sts(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 9 * sizeof(float4));
    }
    __syncthreads();
    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        ptx::stg(C_lds_ptr[i * 36],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

struct FragSub {
    float data[4][4];
    __device__ __forceinline__
    FragSub(const float (&C_frag)[16][8], int prid, int pcid) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[prid * 4 + i][pcid * 4 + j];
            }
        }
    }
};

__device__ __forceinline__
void write_sub_rr(
    FragSub C_sub, uint32_t C_sts_addr, const float *C_lds_ptr, float *C_sub_stg_ptr,
    const uint32_t M, const uint32_t N, uint32_t m_sub_idx, uint32_t n_sub_idx
) {
    __syncthreads();
    #pragma unroll
    for (uint32_t rid = 0; rid < 4; ++rid) {
        ptx::sts(
            C_sub.data[rid][0], C_sub.data[rid][1], C_sub.data[rid][2], C_sub.data[rid][3],
            C_sts_addr + rid * 32 * sizeof(float)
        );
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t iter = 0; iter < 16; ++iter) {
        ptx::stg(
            C_lds_ptr[iter * 32], C_sub_stg_ptr + iter * N,
            (m_sub_idx + iter < M) && (n_sub_idx < N)
        );
    }
}

/**
 * Matrix A, B, C : row-major
 * Threadblock Tile : [M, N, K] = [128, 256, 8]
 * Warp Tile : [M, N, K] = [64, 64, 8]
 * Thread Tile : [M, N, K] = [16, 8, 8]
 * A_tile : [128, 8]
 * B_tile : [8, 256]
 * A_frag : [16, 1]
 * B_frag : [1, 8]
 */
__global__ __launch_bounds__(256)
void ampere_sgemm_rrr_128x256x8_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    // A and B Threadblock Tile on shared memory (double buffer)
    // A_tile : 132 * 8 * float * double buffer = 4.125 KiB * 2
    // B_tile : 256 * 8 * float * double buffer = 8 KiB * 2
    // 为更快地切换 A_tile 和 B_tile 的缓冲区，
    // A_tile 需要一块连续的 8 KiB * 2 = 2^13 B * 2 的缓冲区，故可以使用 (uint32_t&) A_smem ^= 0x2000; 进行切换
    // B_tile 需要一块连续的 8 KiB * 2 = 2^13 B * 2 的缓冲区，故可以使用 (uint32_t&) B_smem ^= 0x2000; 进行切换
    // 如此，共享内存双缓冲的切换，只需要使用一条异或指令即可
    float *smem_buf = buffer::SharedMemory<float, 128 * 8 * 4 + 256 * 8 * 2>().pointer();
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 128 * 8 * 4);

    // A, B Thread Tile on register, C Thread Tile on register (double buffer)
    float A_frag[2][16], B_frag[2][8], C_frag[16][8] = { 0 };

    // A_tile and B_tile ldg pointer, Threadblock arranged as row-major
    // [NEXT] = A_ldg_ptr + K_tile;      [eid] = A_ldg_ptr + eid * 32 * K;                     eid = 0, 1, 2, 3
    // [NEXT] = B_ldg_ptr + K_tile * N;  [eid] = B_ldg_ptr + eid / 2 * 2 * N + eid % 2 * 128;  eid = 0, 1, 2, ..., 7
    const float *A_ldg_ptr = reinterpret_cast<const float*>(A + blockIdx.y * 128 * K + threadIdx.x / 8 * K + threadIdx.x % 8);
    const float *B_ldg_ptr = reinterpret_cast<const float*>(B + blockIdx.x * 256 + threadIdx.x / 128 * N + threadIdx.x % 128);

    // ldg_valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据，避免 ldg 指令越界
    uint32_t A_ldg_valid = 0, B_ldg_valid = 0;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_ldg_valid |= (uint32_t)(blockIdx.y * 128 + threadIdx.x / 8 + eid * 32 < M) << eid;
    }
    for (uint32_t eid = 0; eid < 8; ++eid) {
        B_ldg_valid |= (uint32_t)(blockIdx.x * 256 + threadIdx.x % 128 + eid % 2 * 128 < N) << eid;
    }

    // 一个Warp中的线程标识，排列成 4x8 形状
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t lane_rid = (lane_id / 16) * 2 + (lane_id % 2);
    const uint32_t lane_cid = (lane_id / 2) % 8;

    // A_tile and B_tile sts address
    // [eid] = A_sts_addr + eid * 32 * sizeof(float);                             // eid = 0, 1, 2, 3
    // [eid] = B_sts_addr + (eid / 2 * 2 * 256 + eid % 2 * 128) * sizeof(float);  // eid = 0, 1, 2, ..., 7
    uint32_t A_sts_addr = ptx::smem_addr(A_smem + threadIdx.x % 8 * 132 + threadIdx.x / 8);
    uint32_t B_sts_addr = ptx::smem_addr(B_smem + threadIdx.x / 128 * 256 + threadIdx.x % 128);

    // A_tile and B_tile lds address, 4x2 sub-partitions = [0][0], [0][1], [1][0], [1][1], [2][0], [2][1], [3][0], [3][1]
    // [eid] = A_lds_addr + eid * 132 * sizeof(float);  [prid][pcid] = A_lds_addr + prid * 4 * 4 * sizeof(float);
    // [eid] = B_lds_addr + eid * 256 * sizeof(float);  [prid][pcid] = B_lds_addr + pcid * 8 * 4 * sizeof(float);
    uint32_t A_lds_addr = ptx::smem_addr(A_smem + warp_id / 4 * 64 + lane_rid * 4);
    uint32_t B_lds_addr = ptx::smem_addr(B_smem + warp_id % 4 * 64 + lane_cid * 4);

    // the first A_tile and B_tile load before K-Loop, handling boundary (maybe not 8 data)
    {
        uint32_t first_k_tile = K - ((K + 7) / 8 - 1) * 8;
        #pragma unroll
        for (int eid = 0; eid < 4; ++eid) {
            ptx::ldg_sts(
                A_ldg_ptr + eid * 32 * K,
                A_sts_addr + eid * 32 * sizeof(float),
                threadIdx.x % 8 < first_k_tile ? 4 : 0,
                A_ldg_valid & (1u << eid)
            );
        }
        #pragma unroll
        for (int eid = 0; eid < 8; ++eid) {
            ptx::ldg_sts(
                B_ldg_ptr + eid / 2 * 2 * N + eid % 2 * 128,
                B_sts_addr + (eid / 2 * 2 * 256 + eid % 2 * 128) * sizeof(float),
                threadIdx.x / 128 + eid / 2 * 2 < first_k_tile ? 4 : 0,
                (B_ldg_valid & (1u << eid))
            );
        }
        ptx::ldg_sts_commit();
        __syncthreads();
        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x2000;
        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile;
        B_ldg_ptr += first_k_tile * N;
    }

    // load the first fragment
    ptx::lds(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
    ptx::lds(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7], A_lds_addr + 16 * sizeof(float));
    ptx::lds(A_frag[0][8], A_frag[0][9], A_frag[0][10], A_frag[0][11], A_lds_addr + 32 * sizeof(float));
    ptx::lds(A_frag[0][12], A_frag[0][13], A_frag[0][14], A_frag[0][15], A_lds_addr + 48 * sizeof(float));
    ptx::lds(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
    ptx::lds(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7], B_lds_addr + 32 * sizeof(float));

    // K-Loop, and K_tile is 8
    for (uint32_t num_k_tiles = (K + 7) / 8 - 1; num_k_tiles > 0; --num_k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // K_tile 次计算即将执行完毕，将下一个 A_tile 和 B_tile 写入共享内存
            if (k_frag == 7) {
                ptx::ldg_sts_commit();
                __syncthreads();
                // switch double buffer
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x2000;
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x2000;
                // ldg pointer for next tile
                A_ldg_ptr += 8;
                B_ldg_ptr += 8 * N;
            }
            // 读取下一次计算所需的 A_frag 和 B_frag 并写入寄存器
            ptx::lds(
                A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
                A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
                A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
                A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][8], A_frag[(k_frag + 1) % 2][9],
                A_frag[(k_frag + 1) % 2][10], A_frag[(k_frag + 1) % 2][11],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][12], A_frag[(k_frag + 1) % 2][13],
                A_frag[(k_frag + 1) % 2][14], A_frag[(k_frag + 1) % 2][15],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float)
            );
            // K_tile 的前四次计算之前，读取下一个 A_tile 和 B_tile 数据
            if (k_frag < 4) {
                // for A_tile, eid = k_frag;
                // for B_tild, eid = k_frag + 1;
                ptx::ldg_sts(
                    A_ldg_ptr + k_frag * 32 * K,
                    A_sts_addr + k_frag * 32 * sizeof(float),
                    A_ldg_valid & (1u << k_frag)
                );
                ptx::ldg_sts(
                    B_ldg_ptr + k_frag * 2 * N + 0 * 128,
                    B_sts_addr + (k_frag * 2 * 256 + 0 * 128) * sizeof(float),
                    (B_ldg_valid & (1u << (k_frag * 2)))
                );
                ptx::ldg_sts(
                    B_ldg_ptr + k_frag * 2 * N + 1 * 128,
                    B_sts_addr + (k_frag * 2 * 256 + 1 * 128) * sizeof(float),
                    (B_ldg_valid & (1u << (k_frag * 2 + 1)))
                );
            }
            // 执行FFMA计算
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
                }
            }
        }
    }
    // 最后一个 tile 的迭代
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        // 读取下一次计算所需的 A_frag 和 B_frag 并写入寄存器
        if (k_frag < 7) {
            ptx::lds(
                A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
                A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
                A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
                A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][8], A_frag[(k_frag + 1) % 2][9],
                A_frag[(k_frag + 1) % 2][10], A_frag[(k_frag + 1) % 2][11],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 32) * sizeof(float)
            );
            ptx::lds(
                A_frag[(k_frag + 1) % 2][12], A_frag[(k_frag + 1) % 2][13],
                A_frag[(k_frag + 1) % 2][14], A_frag[(k_frag + 1) % 2][15],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 48) * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 256 * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                B_lds_addr + ((k_frag + 1) % 8 * 256 + 32) * sizeof(float)
            );
        }
        // 执行FFMA计算
        #pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            C_frag[i][j] *= alpha;
        }
    }

    // 以行主序的方式写回矩阵 C_tile 的结果，一行一行地写回，并使用共享内存对数据的布局进行重排，以合并访存
    // 因为一个线程持有 C_frag 数据，而这些数据又划分为 sub-partitions 子分区，每次写回一个子分区 C_sub[4][4] 的数据
    // 在每个线程将 C_sub 写入共享内存时，所使用的地址，共需要 256 * 16 * float 的共享内存空间
    // [rid] = C_sts_addr + rid * 32 * sizeof(float);
    uint32_t C_sts_addr = ptx::smem_addr(smem_buf + warp_id * 32 * 16 + lane_rid * 4 * 32 + lane_cid * 4);
    // 每个线程将共享内存中的数据搬运到全局内存中时，所读取的共享内存的位置 C_lds_ptr 如下
    // 一个 Warp 中的线程负责一个子分区 C_sub_warp[16][32] 的数据，使用 32 个线程搬运 [:][32] 数据，迭代 16 次完成
    // [iter] = C_lds_ptr + iter * 32;
    const float *C_lds_ptr = reinterpret_cast<const float*>(smem_buf + warp_id * 32 * 16 + lane_id);
    // 每个线程要写回的全局内存中的位置，对于一个 C_sub_warp[16][32] 的数据而言，迭代 iter = 16 次完成
    uint32_t m_idx = blockIdx.y * 128 + warp_id / 4 * 64;
    uint32_t n_idx = blockIdx.x * 256 + warp_id % 4 * 64 + lane_id;
    // [prid][pcid][iter] = C_stg_ptr + prid * 16 * N + pcid * 32 + iter * N;
    float *C_stg_ptr = reinterpret_cast<float*>(C + m_idx * N + n_idx);
    // 因为是按行主序的方式写回矩阵 C_tile 的结果，则有效数据行的范围为 [0, 1, 2, ..., M - 1]
    if (m_idx >= M) {
        return;
    } else if (m_idx + 64 > M) {
        // 当前 Warp 所写回的数据存在 M 越界情况，即位于 M 的边界线上
        #pragma unroll
        for (uint32_t prid = 0; prid < 4; ++prid) {
            #pragma unroll
            for (uint32_t pcid = 0; pcid < 2; ++pcid) {
                __syncthreads();
                #pragma unroll
                for (uint32_t rid = 0; rid < 4; ++rid) {
                    ptx::sts(
                        C_frag[prid * 4 + rid][pcid * 4 + 0], C_frag[prid * 4 + rid][pcid * 4 + 1], 
                        C_frag[prid * 4 + rid][pcid * 4 + 2], C_frag[prid * 4 + rid][pcid * 4 + 3], 
                        C_sts_addr + rid * 32 * sizeof(float)
                    );
                }
                __syncthreads();
                #pragma unroll
                for (uint32_t iter = 0; iter < 16; ++iter) {
                    ptx::stg(
                        C_lds_ptr[iter * 32],
                        C_stg_ptr + (prid * 16 + iter) * N + pcid * 32,
                        (m_idx + prid * 16 + iter < M) && (n_idx + pcid * 32 < N)
                    );
                }
            }
        }
    } else /* m_idx + 64 <= M */ {
        // 当前 Warp 所写回的整个全局内存位置都不会越界，无需考虑 M 越界问题，只考虑 N 越界问题
        #pragma unroll
        for (uint32_t prid = 0; prid < 4; ++prid) {
            #pragma unroll
            for (uint32_t pcid = 0; pcid < 2; ++pcid) {
                __syncthreads();
                #pragma unroll
                for (uint32_t rid = 0; rid < 4; ++rid) {
                    ptx::sts(
                        C_frag[prid * 4 + rid][pcid * 4 + 0], C_frag[prid * 4 + rid][pcid * 4 + 1], 
                        C_frag[prid * 4 + rid][pcid * 4 + 2], C_frag[prid * 4 + rid][pcid * 4 + 3], 
                        C_sts_addr + rid * 32 * sizeof(float)
                    );
                }
                __syncthreads();
                #pragma unroll
                for (uint32_t iter = 0; iter < 16; ++iter) {
                    ptx::stg(
                        C_lds_ptr[iter * 32],
                        // 此处 C_sub_stg_ptr 的计算流程应与上处存在差异，以防止编译器将 IMAD 指令密集发射导致占满寄存器
                        C_stg_ptr + prid * 16 * N + pcid * 32 + iter * N,
                        n_idx + pcid * 32 < N
                    );
                }
            }
        }
    }
}

} // sgemm_128x256_16x8