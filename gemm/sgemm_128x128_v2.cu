#pragma once

#include <cuda.h>
#include "../utils/buffer.cu"
#include "../utils/ptx.cu"

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
    const float *A, const float *B, float *C,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    // A and B Threadblock Tile on shared memory (double buffer)
    // A_tile : 132 * 8 * float * double buffer = 4.125 KiB * 2
    // B_tile : 128 * 8 * float * double buffer = 4 KiB * 2
    // 为更快地切换 A_tile 和 B_tile 的缓冲区，
    // A_tile 需要一块连续的 8 KiB * 2 = 2^13 B * 2 的缓冲区，故可以使用 (uint32_t&) A_smem ^= 0x2000; 进行切换
    // B_tile 需要一块连续的 4 KiB * 2 = 2^12 B * 2 的缓冲区，故可以使用 (uint32_t&) B_smem ^= 0x1000; 进行切换
    // 如此，共享内存双缓冲的切换，只需要使用一条异或指令即可
    float *smem_buf = buffer::SharedMemory<float, 128 * 8 * 6>().pointer();
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 128 * 8 * 4);

    // A, B Thread Tile on register, C Thread Tile on register (double buffer)
    float A_frag[2][8], B_frag[2][8], C_frag[8][8] = { 0 };

    // 一个Warp中的线程标识，排列成 4x8 形状
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t lane_rid = (lane_id / 16) * 2 + (lane_id % 2);
    const uint32_t lane_cid = (lane_id / 2) % 8;

    // A_tile and B_tile ldg pointer, Threadblock arranged as row-major
    // [NEXT] = A_ldg_ptr + K_tile;      [eid] = A_ldg_ptr + eid * K;
    // [NEXT] = B_ldg_ptr + K_tile * N;  [eid] = B_ldg_ptr + eid * 32;
    const float *A_ldg_ptr = reinterpret_cast<const float*>(A + blockIdx.y * 128 * K + threadIdx.x / 8 * 4 * K + threadIdx.x % 8);
    const float *B_ldg_ptr = reinterpret_cast<const float*>(B + blockIdx.x * 128 + threadIdx.x / 32 * N + threadIdx.x % 32);

    // A_tile and B_tile sts address
    // [eid] = A_sts_addr + eid * sizeof(float)
    // [eid] = B_sts_addr + eid * 32 * sizeof(float)
    uint32_t A_sts_addr = ptx::smem_addr(A_smem + threadIdx.x % 8 * 132 + threadIdx.x / 8 * 4);
    uint32_t B_sts_addr = ptx::smem_addr(B_smem + threadIdx.x / 32 * 128 + threadIdx.x % 32);

    // A_tile and B_tile lds address, four sub-partitions: [0][0], [0][1], [1][0], [1][1]
    // [eid] = A_lds_addr + eid * 132 * sizeof(float);  [prid][pcid] = A_lds_addr + prid * 4 * 4 * sizeof(float)
    // [eid] = B_lds_addr + eid * 128 * sizeof(float);  [prid][pcid] = B_lds_addr + pcid * 8 * 4 * sizeof(float)
    uint32_t A_lds_addr = ptx::smem_addr(A_smem + warp_id / 2 * 32 + lane_rid * 4);
    uint32_t B_lds_addr = ptx::smem_addr(B_smem + warp_id % 2 * 64 + lane_cid * 4);

    // ldg_valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据，避免 ldg 指令越界
    uint32_t A_ldg_valid = 0, B_ldg_valid = 0;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_ldg_valid |= (uint32_t)(blockIdx.y * 128 + threadIdx.x / 8 * 4 + eid < M)   << eid;
        B_ldg_valid |= (uint32_t)(blockIdx.x * 128 + threadIdx.x % 32 + eid * 32 < N) << eid;
    }

    // A, B ldg buffer for transfering data from gmem to smem
    float A_ldg_buf[4], B_ldg_buf[4];

    // the first A_tile and B_tile load before K-Loop, handling boundary (maybe not 8 data)
    {
        uint32_t first_k_tile = K - ((K + 7) / 8 - 1) * 8;
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem_zero(A_ldg_buf[eid], A_ldg_ptr + eid * K, (A_ldg_valid & (1u << eid)) && threadIdx.x % 8 < first_k_tile);
        }
        ptx::st_smem(A_ldg_buf[0], A_ldg_buf[1], A_ldg_buf[2], A_ldg_buf[3], A_sts_addr);
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::ld_gmem_zero(B_ldg_buf[eid], B_ldg_ptr + eid * 32, (B_ldg_valid & (1u << eid)) && threadIdx.x / 32 < first_k_tile);
        }
        #pragma unroll
        for (uint32_t eid = 0; eid < 4; ++eid) {
            ptx::st_smem(B_ldg_buf[eid], B_sts_addr + eid * 32 * sizeof(float));
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
    ptx::ld_smem(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
    ptx::ld_smem(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7], A_lds_addr + 16 * sizeof(float));
    ptx::ld_smem(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
    ptx::ld_smem(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7], B_lds_addr + 32 * sizeof(float));

    // K-Loop, and K_tile is 8
    for (uint32_t num_k_tiles = (K + 7) / 8 - 1; num_k_tiles > 0; --num_k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // K_tile 次计算即将执行完毕，将下一个 A_tile 和 B_tile 写入共享内存
            if (k_frag == 7) {
                ptx::st_smem(A_ldg_buf[0], A_ldg_buf[1], A_ldg_buf[2], A_ldg_buf[3], A_sts_addr);
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::st_smem(B_ldg_buf[eid], B_sts_addr + eid * 32 * sizeof(float));
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
            ptx::ld_smem(
                A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
                A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
                A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float)
            );
            ptx::ld_smem(
                A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
                A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float)
            );
            ptx::ld_smem(
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float)
            );
            ptx::ld_smem(
                B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float)
            );
            // K_tile 的第一次计算之前，读取下一个 A_tile 和 B_tile 数据
            if (k_frag == 0) {
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::ld_gmem_zero(A_ldg_buf[eid], A_ldg_ptr + eid * K, A_ldg_valid & (1u << eid));
                }
                #pragma unroll
                for (uint32_t eid = 0; eid < 4; ++eid) {
                    ptx::ld_gmem_zero(B_ldg_buf[eid], B_ldg_ptr + eid * 32, B_ldg_valid & (1u << eid));
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
            ptx::ld_smem(
                A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
                A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
                A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float)
            );
            ptx::ld_smem(
                A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
                A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
                A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float)
            );
            ptx::ld_smem(
                B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float)
            );
            ptx::ld_smem(
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

    // 重用 128 * 48 * float 共享内存空间，行主序写回矩阵 C，每次写回一个分区的 16 * 32 * 8 = 128 * 32 * float 数据
    // [trid] = C_sts_addr + trid * 32 * sizeof(float);  trid = 0, 1, 2, 3, for 4x4 Thread Tile
    uint32_t C_sts_addr = ptx::smem_addr(smem_buf + warp_id * 16 * 32 + lane_rid * 4 * 32 + lane_cid * 4);
    // [iter] = C_lds_ptr + iter * 32;  iter = 0, 1, 2, ..., 15, for 16x32 Warp Tile
    const float *C_lds_ptr = reinterpret_cast<const float*>(smem_buf + warp_id * 16 * 32 + lane_id);
    // 将矩阵 C 写回设备内存时的，每个线程对应数据的偏移
    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;
    // [prid][pcid][iter] = C_stg_ptr + prid * 16 * N + pcid * 32 + iter * N
    float *C_stg_ptr = reinterpret_cast<float*>(C + m_idx * N + n_idx);
    #pragma unroll
    for (uint32_t prid = 0; prid < 2; ++prid) {
        #pragma unroll
        for (uint32_t pcid = 0; pcid < 2; ++pcid) {
            __syncthreads();
            #pragma unroll
            for (uint32_t trid = 0; trid < 4; ++trid) {
                ptx::st_smem(
                    C_frag[prid * 4 + trid][pcid * 4 + 0], C_frag[prid * 4 + trid][pcid * 4 + 1],
                    C_frag[prid * 4 + trid][pcid * 4 + 2], C_frag[prid * 4 + trid][pcid * 4 + 3],
                    C_sts_addr + trid * 32 * sizeof(float)
                );
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t iter = 0; iter < 16; ++iter) {
                ptx::st_gmem(
                    C_lds_ptr[iter * 32], C_stg_ptr + prid * 16 * N + pcid * 32 + iter * N,
                    (m_idx + prid * 16 + iter < M) && (n_idx + pcid * 32 < N)
                );
            }
        }
    }
}

__device__ __forceinline__
void store_result_smem_rr(
    float Creg[8][8], float *smem_buf, float *C,
    const uint32_t M, const uint32_t N, const uint32_t cS,
    const uint32_t brid, const uint32_t bcid, const uint32_t tid,
    const uint32_t wrows, const uint32_t wcols, const uint32_t wrid, const uint32_t wcid,
    const uint32_t lrid, const uint32_t lcid
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

void sgemm_rrr(
    const float *A, const float *B, float *C,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 127) / 128, (M + 127) / 128, 1);
    sgemm_rrr_128x128x8_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}