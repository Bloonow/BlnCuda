#pragma once

#include <cuda.h>
#include "../utils/buffer.cu"
#include "../utils/ptx.cu"

namespace sgemm_128x128_8x8 {

/**
 * Matrix A, B, C : row-major
 * Threadblock Tile : [M, N, K] = [128, 128, 8]
 * Warp Tile : [M, N, K] = [32, 64, 8]
 * Thread Tile : [M, N, K] = [8, 8, 8]
 * A_tile and B_tile : [128, 8] and [8, 128]
 * A_frag and B_frag : [8, 1] and [1, 8]
 */
__global__ __launch_bounds__(256, 2)
void sgemm_rrr_128x128x8_kernel(
    const float *A, const float *B, float *C, const float alpha,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    // A and B Threadblock Tile on shared memory (double buffer)
    // A_tile : 132 * 8 * float * double buffer = 4.125 KiB * 2
    // B_tile : 128 * 8 * float * double buffer = 4 KiB * 2
    // 为更快地切换 A_tile 和 B_tile 的缓冲区，
    // A_tile 需要一块连续的 8 KiB * 2 = 2^13 B * 2 的缓冲区，故可以使用 (uint32_t&) A_smem ^= 0x2000; 进行切换
    // B_tile 需要一块连续的 4 KiB * 2 = 2^12 B * 2 的缓冲区，故可以使用 (uint32_t&) B_smem ^= 0x1000; 进行切换
    // 如此，共享内存双缓冲的切换，只需要使用一条异或指令即可
    float *smem_buf = buffer::SharedMemory<float, 128 * 8 * 4 + 128 * 8 * 2>().pointer();
    float *A_smem = smem_buf;
    float *B_smem = smem_buf + 128 * 8 * 4;

    // A, B Thread Tile on register, C Thread Tile on register (double buffer)
    float A_frag[2][8], B_frag[2][8], C_frag[8][8] = { 0 };

    // A_tile and B_tile ldg pointer, Threadblock arranged as row-major
    // [NEXT] = A_ldg_ptr + K_tile;      [eid] = A_ldg_ptr + eid * K;
    // [NEXT] = B_ldg_ptr + K_tile * N;  [eid] = B_ldg_ptr + eid * 32;
    const float *A_ldg_ptr = A + blockIdx.y * 128 * K + threadIdx.x / 8 * 4 * K + threadIdx.x % 8;
    const float *B_ldg_ptr = B + blockIdx.x * 128 + threadIdx.x / 32 * N + threadIdx.x % 32;

    // ldg_valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据，避免 ldg 指令越界
    uint32_t A_ldg_valid = 0, B_ldg_valid = 0;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_ldg_valid |= (uint32_t)(blockIdx.y * 128 + threadIdx.x / 8 * 4 + eid < M)   << eid;
        B_ldg_valid |= (uint32_t)(blockIdx.x * 128 + threadIdx.x % 32 + eid * 32 < N) << eid;
    }

    // 一个Warp中的线程标识，排列成 4x8 形状
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t lane_rid = (lane_id / 16) * 2 + (lane_id % 2);
    const uint32_t lane_cid = (lane_id / 2) % 8;

    // A_tile and B_tile sts address
    // [eid] = A_sts_addr + eid * sizeof(float);
    // [eid] = B_sts_addr + eid * 32 * sizeof(float);
    uint32_t A_sts_addr = ptx::smem_addr(A_smem + threadIdx.x % 8 * 132 + threadIdx.x / 8 * 4);
    uint32_t B_sts_addr = ptx::smem_addr(B_smem + threadIdx.x / 32 * 128 + threadIdx.x % 32);

    // A_tile and B_tile lds address, 2x2 sub-partitions = [0][0], [0][1], [1][0], [1][1]
    // [eid] = A_lds_addr + eid * 132 * sizeof(float);  [prid][pcid] = A_lds_addr + prid * 4 * 4 * sizeof(float);
    // [eid] = B_lds_addr + eid * 128 * sizeof(float);  [prid][pcid] = B_lds_addr + pcid * 8 * 4 * sizeof(float);
    uint32_t A_lds_addr = ptx::smem_addr(A_smem + warp_id / 2 * 32 + lane_rid * 4);
    uint32_t B_lds_addr = ptx::smem_addr(B_smem + warp_id % 2 * 64 + lane_cid * 4);

    // A, B ldg buffer for transfering data from gmem to smem
    float A_ldg_buf[4], B_ldg_buf[4];

    // the first A_tile and B_tile load before K-Loop, handling boundary (maybe not 8 data)
    {
        uint32_t first_k_tile = K - ((K + 7) / 8 - 1) * 8;
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ldg_zero(A_ldg_buf[eid], A_ldg_ptr + eid * K, (A_ldg_valid & (1u << eid)) && (threadIdx.x % 8 < first_k_tile));
        }
        ptx::sts(A_ldg_buf[0], A_ldg_buf[1], A_ldg_buf[2], A_ldg_buf[3], A_sts_addr);
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ldg_zero(B_ldg_buf[eid], B_ldg_ptr + eid * 32, (B_ldg_valid & (1u << eid)) && (threadIdx.x / 32 < first_k_tile));
        }
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::sts(B_ldg_buf[eid], B_sts_addr + eid * 32 * sizeof(float));
        }
        __syncthreads();
        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x1000;
        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile;
        B_ldg_ptr += first_k_tile * N;
    }

    // load the first fragment
    ptx::lds(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
    ptx::lds(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7], A_lds_addr + 16 * sizeof(float));
    ptx::lds(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
    ptx::lds(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7], B_lds_addr + 32 * sizeof(float));

    // K-Loop, and K_tile is 8
    for (uint32_t num_k_tiles = (K + 7) / 8 - 1; num_k_tiles > 0; --num_k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // K_tile 次计算即将执行完毕，将下一个 A_tile 和 B_tile 写入共享内存
            if (k_frag == 7) {
                ptx::sts(A_ldg_buf[0], A_ldg_buf[1], A_ldg_buf[2], A_ldg_buf[3], A_sts_addr);
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::sts(B_ldg_buf[eid], B_sts_addr + eid * 32 * sizeof(float));
                }
                __syncthreads();
                // switch double buffer
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
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
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float)
            );
            // K_tile 的第一次计算之前，读取下一个 A_tile 和 B_tile 数据
            if (k_frag == 0) {
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::ldg_zero(A_ldg_buf[eid], A_ldg_ptr + eid * K, A_ldg_valid & (1u << eid));
                }
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::ldg_zero(B_ldg_buf[eid], B_ldg_ptr + eid * 32, B_ldg_valid & (1u << eid));
                }
            }
            // 执行FFMA计算
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
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
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float)
            );
            ptx::lds(
                B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float)
            );
        }
        // 执行FFMA计算
        #pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
            }
        }
    }
    // 应用 alpha 缩放
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            C_frag[i][j] *= alpha;
        }
    }

    // 以行主序的方式写回矩阵 C_tile 的结果，一行一行地写回，并使用共享内存对数据的布局进行重排，以合并访存
    // 因为一个线程持有 C_frag 数据，而这些数据又划分为 sub-partitions 子分区，每次写回一个子分区 C_sub[4][4] 的数据
    // 在每个线程将 C_sub 写入共享内存时，共需要 256 * 16 * float 的共享内存空间
    // [rid] = C_sts_addr + rid * 32 * sizeof(float);
    uint32_t C_sts_addr = ptx::smem_addr(smem_buf + warp_id * 32 * 16 + lane_rid * 4 * 32 + lane_cid * 4);
    // 每个线程将共享内存中的数据搬运到全局内存中时，所读取的共享内存的位置 C_lds_ptr 如下
    // 一个 Warp 中的线程负责一个子分区 C_sub_warp[16][32] 的数据，使用 32 个线程搬运 [:][32] 数据，迭代 16 次完成
    // [iter] = C_lds_ptr + iter * 32;
    const float *C_lds_ptr = smem_buf + warp_id * 32 * 16 + lane_id;
    // 每个线程要写回的全局内存中的位置，对于一个 C_sub_warp[16][32] 的数据而言，迭代 iter = 16 次完成
    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;
    // [prid][pcid][iter] = C_stg_ptr + prid * 16 * N + pcid * 32 + iter * N;
    float *C_stg_ptr = C + m_idx * N + n_idx;
    // 因为是按行主序的方式写回矩阵 C_tile 的结果，则有效数据行的范围为 [0, 1, 2, ..., M - 1]
    if (m_idx >= M) {
        return;
    } else if (m_idx + 32 > M) {
        // 当前 Warp 所写回的数据存在 M 越界情况，即位于 M 的边界线上
        #pragma unroll
        for (uint32_t prid = 0; prid < 2; ++prid) {
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
    } else /* m_idx + 32 <= M */ {
        // 当前 Warp 所写回的整个全局内存位置都不会越界，无需考虑 M 越界问题，只考虑 N 越界问题
        #pragma unroll
        for (uint32_t prid = 0; prid < 2; ++prid) {
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

} // namespace sgemm_128x128_8x8