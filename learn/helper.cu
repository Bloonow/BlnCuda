/**
 * All vectors and matrices store in column-major format.
 * 🤣 👉 🤡
 * Matrices can store both in row-major and column-major now.
*/

#include <cuda.h>
#include <time.h>
#include <stdarg.h>

template<typename Ty>
Ty* alloc_host_memory(size_t count, Ty max_init = (Ty)(1)) {
    static time_t init_time = 0;
    if (init_time == 0) srand(time(&init_time));
    Ty* ptr = reinterpret_cast<Ty*>(malloc(sizeof(Ty) * count));
    for (size_t i = 0; i < count; i++) {
        ptr[i] = static_cast<Ty>(static_cast<double>(rand()) / RAND_MAX * max_init);
    }
    return ptr;
}

template<typename Ty>
Ty* alloc_cuda_memory(size_t count, Ty *host_ptr = nullptr) {
    Ty* ptr;
    cudaMalloc(&ptr, sizeof(Ty) * count);
    if (host_ptr != nullptr) {
        cudaMemcpy(ptr, host_ptr, sizeof(Ty) * count, cudaMemcpyHostToDevice);
    }
    return ptr;
}

void free_memory(size_t count, ...) {
    va_list vlist;
    va_start(vlist, count);
    for (size_t i = 0; i < count; i++) {
        void *ptr = va_arg(vlist, void*);
        // try free as gpu pointer
        cudaError_t status = cudaFree(ptr);
        // free as host pointer
        if (status != 0) free(ptr);
    }
    va_end(vlist);
}

template<typename Ty>
bool check_same(Ty *ptr1, Ty *ptr2, size_t num, Ty error = 1e-4) {
    for (size_t i = 0; i < num; i++) {
        if (abs(ptr1[i] - ptr2[i]) > error) return false;
    }
    return true;
}

typedef enum Order {
    ROW_MAJOR = 0,
    COL_MAJOR = 1
} Order_t;

inline size_t row_index(size_t rid, size_t cid, size_t rows, size_t cols) {
    return rid * cols + cid;
}

inline size_t col_index(size_t rid, size_t cid, size_t rows, size_t cols) {
    return cid * rows + rid;
}

template<typename Ty>
void host_gemv(
    size_t M, size_t N, Order_t A_order, Ty *A, Ty *x, Ty *y, Ty alpha, Ty beta, size_t batch_count
) {
    auto A_idx = A_order == ROW_MAJOR ? row_index : col_index;
    for (size_t bid = 0; bid < batch_count; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            Ty value = 0;
            for (size_t cid = 0; cid < N; cid++) {
                value += A[bid * M * N + A_idx(rid, cid, M, N)] * x[bid * N + cid];
            }
            y[bid * M + rid] = alpha * value + beta * y[bid * M + rid];
        }
    }
}

template<typename Ty>
void host_gemm(
    size_t M, size_t N, size_t K, Order_t A_order, Order_t B_order, Order_t C_order,
    Ty *A, Ty *B, Ty *C, Ty alpha, Ty beta, size_t batch_count
) {
    auto A_idx = A_order == ROW_MAJOR ? row_index : col_index;
    auto B_idx = B_order == ROW_MAJOR ? row_index : col_index;
    auto C_idx = C_order == ROW_MAJOR ? row_index : col_index;
    for (size_t bid = 0; bid < batch_count; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            for (size_t cid = 0; cid < N; cid++) {
                Ty value = 0;
                for (size_t k = 0; k < K; k++) {
                    value += A[bid * M * K + A_idx(rid, k, M, K)] * B[bid * K * N + B_idx(k, cid, K, N)];
                }
                C[bid * M * N + C_idx(rid, cid, M, N)] =
                    alpha * value + beta * C[bid * M * N + C_idx(rid, cid, M, N)];
            }
        }
    }
}