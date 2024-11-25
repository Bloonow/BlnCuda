#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

#include "../utils/buffer.cu"
#include "../utils/ptx.cu"

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}

__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
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
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */

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
    float *smem_buf = buffer::SharedMemory<float, 128 * 8 * 6>().pointer();
    float *A_smem = reinterpret_cast<float*>(smem_buf);
    float *B_smem = reinterpret_cast<float*>(smem_buf + 128 * 8 * 4);

    // A, B ldg buffer for transfering data from gmem to smem
    float A_ldg_buf[4], B_ldg_buf[4];

    // A, B Thread Tile on register, C Thread Tile on register (double buffer)
    float A_frag[2][8], B_frag[2][8], C_frag[8][8] = { 0 };

    // A_tile and B_tile ldg pointer, Threadblock arranged as row-major
    // [NEXT] = A_ldg_ptr + K_tile;      [eid] = A_ldg_ptr + eid * K;
    // [NEXT] = B_ldg_ptr + K_tile * N;  [eid] = B_ldg_ptr + eid * 32;
    const float *A_ldg_ptr = reinterpret_cast<const float*>(A + blockIdx.y * 128 * K + threadIdx.x / 8 * 4 * K + threadIdx.x % 8);
    const float *B_ldg_ptr = reinterpret_cast<const float*>(B + blockIdx.x * 128 + threadIdx.x / 32 * N + threadIdx.x % 32);

    // ldg_valid[eid] 标识 eid 数据是否为有效数据，有效元素指未越界的数据，避免 ldg 指令越界
    uint32_t A_ldg_valid = 0, B_ldg_valid = 0;
    #pragma unroll
    for (uint32_t eid = 0; eid < 4; ++eid) {
        A_ldg_valid |= (uint32_t)(blockIdx.y * 128 + threadIdx.x / 8 * 4 + eid < M)   << eid;
        B_ldg_valid |= (uint32_t)(blockIdx.x * 128 + threadIdx.x % 32 + eid * 32 < N) << eid;
    }

    // original : ·-→ x   now :  ·-→ cid
    //            ↓              ↓
    //            y             rid
    // 一个Warp中的线程标识，排列成 4x8 形状
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t /* mma_tid_y */ lane_rid = (lane_id / 16) * 2 + (lane_id % 2);
    const uint32_t /* mma_tid_x */ lane_cid = (lane_id / 2) % 8;

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
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7], A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7], B_lds_addr + 32 * sizeof(float));

    // K-Loop
    for (uint32_t num_k_tiles = (K + 7) / 8 - 1; num_k_tiles > 0; --num_k_tiles) {
        ?
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step;
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0);
                }

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 127) / 128, (m + 127) / 128);

    // warmup
    sgemm_128x128x8_kernel<<<grid, 256>>>(
        d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        sgemm_128x128x8_kernel<<<grid, 256>>>(
            d_A, d_B, d_C, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}


