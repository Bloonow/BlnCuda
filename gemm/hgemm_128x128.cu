#pragma once

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include "../utils/buffer.cu"

namespace wmma_hgemm_m16n16k16 {
using namespace nvcuda;

/* [WHEN] SM's Shared Memory > 72KiB */
struct ShapeLayout {
    static constexpr uint32_t M_wmma = 16;
    static constexpr uint32_t N_wmma = 16;
    static constexpr uint32_t K_wmma = 16;
    static constexpr uint32_t M_tile = 128;
    static constexpr uint32_t N_tile = 128;
    static constexpr uint32_t K_tile = 128;
    static constexpr uint32_t Chunks_K = 8;
    static constexpr uint32_t Skews_half = 16;
    static constexpr uint32_t M_warp = 32;
    static constexpr uint32_t N_warp = 64;
    static constexpr uint32_t Block_warp_rows = 4;
    static constexpr uint32_t Block_warp_cols = 2;
    static constexpr uint32_t Warp_wmma_rows = 2;
    static constexpr uint32_t Warp_wmma_cols = 4;
    uint32_t brid, bcid, wid, lid, wrid, wcid;
    __device__ ShapeLayout() {
        brid = blockIdx.y;
        bcid = blockIdx.x;
        wid = threadIdx.x / 32;
        lid = threadIdx.x % 32;
        wrid = wid / Block_warp_cols;
        wcid = wid % Block_warp_cols;
    }
};

/**
 * Matrix A, B, C : row-major, col-major, row-major
 */
__global__ void wmma_hgemm_m16n16k16_128x128x128_kernel(
    const half* A, const half* B, const float* C, float* D, float alpha, float beta,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    extern __shared__ void* smem_buf[];
    ShapeLayout SL;

    // 一个线程块共128x128的数据，由8个Warp负责，一个Warp负责32x64的数据，这些Warp排列成4x2的形状
    // 一个Warp共32x64的数据，可以分成8个wmma块，一个wmma负责16x16的数据，这些wmma排列成2x4的形状
    // [wmma_rid][wmma_cid] = acc_smem_ptr + wmma_rid * 16 * 128 + wmma_cid * 16;
    float* acc_smem_ptr = reinterpret_cast<float*>(smem_buf) + SL.wrid * 32 * 128 + SL.wcid * 64;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[2][4];
    #pragma unroll
    for (uint32_t rid = 0; rid < 2; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            wmma::fill_fragment(acc_frag[rid][cid], 0);
        }
    }

    if (C != nullptr && beta != 0) {
        // 直接将 beta * C 加到最终结果 acc_frag 上，如此无需再占用额外的 c_frag 寄存器
        beta /= alpha;
        // 一个线程一次性最多加载16字节也即4个float数据，一个Warp一次性加载128个float数据，正好是一行N_tile个float数据
        // 于是，一共8个Warp可以一次性加载8行的float数据，则需迭代16次才能加载完128行的float数据
        // [iter] = C_sts_ptr + iter * 8 * 128;
        float* C_sts_ptr = reinterpret_cast<float*>(smem_buf) + SL.wid * 128 + SL.lid * 4;
        // [iter] = C_ldg_ptr + iter * 8 * N;
        const float* C_ldg_ptr = C + SL.brid * 128 * N + SL.bcid * 128 + SL.wid * N + SL.lid * 4;
        #pragma unroll
        for (uint32_t iter = 0; iter < 16; ++iter) {
            *reinterpret_cast<uint4*>(C_sts_ptr + iter * 8 * 128) = *reinterpret_cast<const uint4*>(C_ldg_ptr + iter * 8 * N);
        }
        __syncthreads();
        // 使用wmma::load_matrix_sync将数据从共享内存加载到acc_frag寄存器当中
        #pragma unroll
        for (uint32_t rid = 0; rid < 2; ++rid) {
            #pragma unroll
            for (uint32_t cid = 0; cid < 4; ++cid) {
                wmma::load_matrix_sync(acc_frag[rid][cid], acc_smem_ptr + rid * 16 * 128 + cid * 16, 128, wmma::mem_row_major);
                #pragma unroll
                for (uint32_t i = 0; i < acc_frag[rid][cid].num_elements; ++i) {
                    acc_frag[rid][cid].x[i] *= beta;
                }
            }
        }
        __syncthreads();
    }

    // A_tile[M_tile][K_tile] --> A_smem[M_tile][K_tile + Skews_K]
    half* A_smem = reinterpret_cast<half*>(smem_buf);
    // B_tile[N_tile][K_tile] --> B_smem[N_tile][K_tile + Skews_K]
    half* B_smem = reinterpret_cast<half*>(smem_buf) + 128 * (128 + 16);

    // 一个线程一次性最多加载16字节也即8个half数据，一个Warp一次性加载256个half数据，正好是2行N_tile个hlaf数据
    // 前0~15号线程加载第一行half数据，后16~31号线程加载第二行half数据，注意每行存在16个half的偏移
    // 于是，一共8个Warp可以一次性加载16行的half数据，则需迭代8次才能加载完128行的half数据
    // [iter] = [A|B]_sts_ptr + iter * 16 * 144;
    half* A_sts_ptr = A_smem + SL.wid * 2 * 144 + SL.lid / 16 * 144 + SL.lid % 16 * 8;
    half* B_sts_ptr = B_smem + SL.wid * 2 * 144 + SL.lid / 16 * 144 + SL.lid % 16 * 8;
    // [iter] = [A|B]_ldg_ptr + iter * 16 * K;  [NEXT] = [A|B]_ldg_ptr + K_tile;
    const half* A_ldg_ptr = A + SL.brid * 128 * K + SL.wid * 2 * K + SL.lid / 16 * K + SL.lid % 16 * 8;
    const half* B_ldg_ptr = B + SL.bcid * 128 * K + SL.wid * 2 * K + SL.lid / 16 * K + SL.lid % 16 * 8;

    // 一个线程块共128x128的数据，由8个Warp负责，一个Warp负责32x64的数据，这些Warp排列成4x2的形状
    // 一个Warp共32x64的数据，可以分成8个wmma块，一个wmma负责16x16的数据，这些wmma排列成2x4的形状
    // [chunk][wmma_rid][wmma_cid] = A_lds_ptr + chunk * 16 + wmma_rid * 16 * 144;
    // [chunk][wmma_rid][wmma_cid] = B_lds_ptr + chunk * 16 + wmma_cid * 16 * 144;
    const half* A_lds_ptr = A_smem + SL.wrid * 32 * 144;
    const half* B_lds_ptr = B_smem + SL.wcid * 64 * 144;

    // K-Loop, and K_tile = Chunks_K * K_wmma = 8 * 16 = 128
    for (uint32_t num_k_tiles = K / 128; num_k_tiles > 0; --num_k_tiles) {
        // load A_tile and B_tile from global memory to A_smem and B_smem
        #pragma unroll
        for (uint32_t iter = 0; iter < 8; ++iter) {
            *reinterpret_cast<uint4*>(A_sts_ptr + iter * 16 * 144) = *reinterpret_cast<const uint4*>(A_ldg_ptr + iter * 16 * K);
            *reinterpret_cast<uint4*>(B_sts_ptr + iter * 16 * 144) = *reinterpret_cast<const uint4*>(B_ldg_ptr + iter * 16 * K);
        }
        __syncthreads();
        // ldg pointer for next tile
        A_ldg_ptr += 128;
        B_ldg_ptr += 128;
        // 执行WMMA计算
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];
            #pragma unroll
            for (uint32_t rid = 0; rid < 2; ++rid) {
                wmma::load_matrix_sync(a_frag[rid], A_lds_ptr + chunk * 16 + rid * 16 * 144, 144);
                #pragma unroll
                for (uint32_t cid = 0; cid < 4; ++cid) {
                    if (rid == 0) {
                        // 因为wmma排列成2x4的形状，第一行和第二行使用一样的b_frag数据，故只加载一次即可
                        wmma::load_matrix_sync(b_frag[cid], B_lds_ptr + chunk * 16 + cid * 16 * 144, 144);
                    }
                    wmma::mma_sync(acc_frag[rid][cid], a_frag[rid], b_frag[cid], acc_frag[rid][cid]);
                }
            }
        }
        __syncthreads();
    }

    // 将计算得到的acc_frag结果写回共享内存当中
    #pragma unroll
    for (uint32_t rid = 0; rid < 2; ++rid) {
        #pragma unroll
        for (uint32_t cid = 0; cid < 4; ++cid) {
            // Use the alpha to scale acc_frag
            #pragma unroll
            for (uint32_t i = 0; i < acc_frag[rid][cid].num_elements; i++) { acc_frag[rid][cid].x[i] *= alpha; }
            wmma::store_matrix_sync(acc_smem_ptr + rid * 16 * 128 + cid * 16, acc_frag[rid][cid], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 一个线程一次性最多存储16字节也即4个float数据，一个Warp一次性存储128个float数据，正好是一行N_tile个float数据
    // 于是，一共8个Warp可以一次性存储8行的float数据，则需迭代16次才能存储完128行的float数据
    // [iter] = D_lds_ptr + iter * 8 * 128;
    float* D_lds_ptr = reinterpret_cast<float*>(smem_buf) + SL.wid * 128 + SL.lid * 4;
    // [iter] = D_stg_ptr + iter * 8 * N;
    float* D_stg_ptr = D + SL.brid * 128 * N + SL.bcid * 128 + SL.wid * N + SL.lid * 4;
    #pragma unroll
    for (uint32_t iter = 0; iter < 16; ++iter) {
        *reinterpret_cast<uint4*>(D_stg_ptr + iter * 8 * N) = *reinterpret_cast<const uint4*>(D_lds_ptr + iter * 8 * 128);
    }
}

__host__ void wmma_hgemm_rcr_cuda(
    const half* A, const half* B, const float* C, float* D, float alpha, float beta,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);
    assert(M % 128 == 0 && N % 128 == 0 && K % 128 == 0);
    const dim3 block_size(256, 1, 1);
    const dim3 grid_size((N + 127) / 128, (M + 127) / 128, 1);
    uint32_t smem_bytes = max(128 * (128 + 16) * 2 * sizeof(half), 128 * 128 * sizeof(float));
    cudaFuncSetAttribute(
        wmma_hgemm_m16n16k16_128x128x128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes
    );
    wmma_hgemm_m16n16k16_128x128x128_kernel<<<grid_size, block_size, smem_bytes>>>(
        A, B, C, D, alpha, beta, M, N, K
    );
}

/**
 * Matrix A, B, C : row-major, col-major, row-major
 * Threadblock Tile : [M, N, K] = [64, 64, 16]
 * Warp Tile : [M, N, K] = [16, 16, 16]
 */
__global__ void simple_wmma_hgemm_m16n16k16_64x64x16_kernel(
    const half* A, const half* B, const float* C, float* D, float alpha, float beta,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    uint32_t warp_rid_offset = blockIdx.y * 64 + threadIdx.x / 32 / 4 * 16;
    uint32_t warp_cid_offset = blockIdx.x * 64 + threadIdx.x / 32 % 4 * 16;

    // [A|B][NEXT] = [A|B]_ldg_ptr + K_tile;  A is row-major, B is col-major
    const half* A_ldg_ptr = A + warp_rid_offset * K;
    const half* B_ldg_ptr = B + warp_cid_offset * K;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0);

    // Bounds checking
    if (warp_rid_offset < M && warp_cid_offset < N) {
        if (C != nullptr && beta != 0) {
            // 直接将 beta * C 加到最终结果 acc_frag 上，如此无需再占用额外的 c_frag 寄存器
            // 为保证最终结果乘以 alpha 之后正确，此处对 beta 值进行缩放修正
            beta /= alpha;
            const float* C_ldg_ptr = C + warp_rid_offset * N + warp_cid_offset;
            wmma::load_matrix_sync(acc_frag, C_ldg_ptr, N, wmma::mem_row_major);
            #pragma unroll
            for (uint32_t i = 0; i < acc_frag.num_elements; i++) { acc_frag.x[i] *= beta; }
        }

        // K-Loop, and K_tile is 16
        for (uint32_t num_k_tiles = K / 16; num_k_tiles > 0; --num_k_tiles) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A_ldg_ptr, K);
            wmma::load_matrix_sync(b_frag, B_ldg_ptr, K);
            // ldg pointer for next tile
            A_ldg_ptr += 16;
            B_ldg_ptr += 16;
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        // Use the alpha to scale acc_frag
        #pragma unroll
        for (uint32_t i = 0; i < acc_frag.num_elements; i++) { acc_frag.x[i] *= alpha; }
        // Store the output
        float* D_stg_ptr = D + warp_rid_offset * N + warp_cid_offset;
        wmma::store_matrix_sync(D_stg_ptr, acc_frag, N, wmma::mem_row_major);
    }
}

__host__ void simple_wmma_hgemm_rcr_cuda(
    const half* A, const half* B, const float* C, float* D, float alpha, float beta,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    assert(((unsigned long long)A) % 64 == 0);
    assert(((unsigned long long)B) % 64 == 0);
    assert(((unsigned long long)C) % 64 == 0);
    assert(((unsigned long long)D) % 64 == 0);
    assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);
    const dim3 block_size(512, 1, 1);
    const dim3 grid_size((N + 63) / 64, (M + 63) / 64, 1);
    simple_wmma_hgemm_m16n16k16_64x64x16_kernel<<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);
}

} // namespace wmma_hgemm_m16n16k16