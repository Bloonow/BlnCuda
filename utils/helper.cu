#include <cuda.h>
#include <cuda_fp16.hpp>
#include <time.h>
#include <stdarg.h>
#include <stdio.h>

template<typename Ty>
Ty* alloc_host_memory(size_t count, double max_init = 1) {
    static time_t init_time = 0;
    if (init_time == 0) srand(time(&init_time));
    Ty *ptr = reinterpret_cast<Ty*>(malloc(sizeof(Ty) * count));
    for (size_t i = 0; i < count; i++) {
        double value = rand() % 2 == 0 ? 1 : -1;
        value *= static_cast<double>(rand()) / RAND_MAX * max_init;
        ptr[i] = static_cast<Ty>(value);
    }
    return ptr;
}

template<typename Ty>
Ty* alloc_cuda_memory(size_t count, Ty *host_ptr = nullptr) {
    Ty *ptr;
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
        // free as cuda pointer
        cudaError_t status = cudaFree(ptr);
        // free as host pointer
        if (status != cudaSuccess) free(ptr);
    }
    va_end(vlist);
}

template<typename Ty>
bool check_same(Ty *ptr1, Ty *ptr2, size_t count, Ty error = 1e-5) {
    bool same = true;
    printf("🔵 Data same checking... 🧐🧐🧐 --> between [%p] and [%p] address.\n", ptr1, ptr2);
    for (size_t i = 0; i < count; i++) {
        if (abs(ptr1[i] - ptr2[i]) >= error) {
            printf("🔴 Data are not same! 🤡🤡🤡 --> ptr1[%ld] = %lf != %lf = ptr2[%ld]\n", i, ptr1[i], ptr2[i], i);
            same = false;
            break;
        }
    }
    if (same) printf("🟢 All data are same! 💃💃💃\n");
    return same;
}

// Tags
struct row_major;
struct col_major;

template<typename Ty, typename layout> struct Accessor;

template<typename Ty>
struct Accessor<Ty, row_major> {
    Ty* data;
    size_t rows, cols;
    Accessor(Ty* data_, size_t rows_, size_t cols_) : data(data_), rows(rows_), cols(cols_) {}
    Accessor(Ty* data_, size_t cols_) : data(data_), rows(1), cols(cols_) {}
    Ty& operator()(size_t bid, size_t rid, size_t cid) {
        return data[bid * rows * cols + rid * cols + cid];
    }
    Ty& operator()(size_t bid, size_t cid) {
        return data[bid * cols + cid];
    }
};

template<typename Ty>
struct Accessor<Ty, col_major> {
    Ty* data;
    size_t rows, cols;
    Accessor(Ty* data_, size_t rows_, size_t cols_) : data(data_), rows(rows_), cols(cols_) {}
    Accessor(Ty* data_, size_t rows_) : data(data_), rows(rows_), cols(1) {}
    Ty& operator()(size_t bid, size_t rid, size_t cid) {
        return data[bid * rows * cols + cid * rows + rid];
    }
    Ty& operator()(size_t bid, size_t rid) {
        return data[bid * rows + rid];
    }
};

template<typename Ty, typename layout>
void host_gemv(
    Ty* A, Ty* x, Ty* y, Ty alpha, Ty beta,
    size_t M, size_t N, size_t batch_size
) {
    Accessor<Ty, layout> A_ = Accessor<Ty, layout>(A, M, N);
    Accessor<Ty, col_major> x_ = Accessor<Ty, col_major>(x, N);
    Accessor<Ty, col_major> y_ = Accessor<Ty, col_major>(y, M);
    for (size_t bid = 0; bid < batch_size; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            Ty value = 0;
            for (size_t kid = 0; kid < N; kid++) {
                value += A_(bid, rid, kid) * x_(bid, kid);
            }
            y_(bid, rid) = alpha * value + beta * y_(bid, rid);
        }
    }
}

template<typename Ty, typename A_layout, typename B_layout, typename C_layout>
void host_gemm(
    Ty* A, Ty* B, Ty* C, Ty alpha, Ty beta,
    size_t M, size_t N, size_t K, size_t batch_size
) {
    Accessor<Ty, A_layout> A_ = Accessor<Ty, A_layout>(A, M, K);
    Accessor<Ty, B_layout> B_ = Accessor<Ty, B_layout>(B, K, N);
    Accessor<Ty, C_layout> C_ = Accessor<Ty, C_layout>(C, M, N);
    for (size_t bid = 0; bid < batch_size; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            for (size_t cid = 0; cid < N; cid++) {
                Ty value = 0;
                for (size_t kid = 0; kid < K; kid++) {
                    value += A_(bid, rid, kid) * B_(bid, kid, cid);
                }
                C_(bid, rid, cid) = alpha * value + beta * C_(bid, rid, cid);
            }
        }
    }
}

// specialization for half which supports '*' just on device
template<typename A_layout, typename B_layout, typename C_layout>
void host_gemm(
    half* A, half* B, float* C, float alpha, float beta,
    size_t M, size_t N, size_t K, size_t batch_size
) {
    Accessor<half, A_layout> A_ = Accessor<half, A_layout>(A, M, K);
    Accessor<half, B_layout> B_ = Accessor<half, B_layout>(B, K, N);
    Accessor<float, C_layout> C_ = Accessor<float, C_layout>(C, M, N);
    for (size_t bid = 0; bid < batch_size; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            for (size_t cid = 0; cid < N; cid++) {
                float value = 0;
                for (size_t kid = 0; kid < K; kid++) {
                    value += __half2float(A_(bid, rid, kid)) * __half2float(B_(bid, kid, cid));
                }
                C_(bid, rid, cid) = alpha * value + beta * C_(bid, rid, cid);
            }
        }
    }
}

template<typename Ty, typename A_layout, typename B_layout, typename C_layout, typename D_layout>
void host_matmul_relu(
    Ty* A, Ty* B, Ty* C, Ty* D, Ty* bias, Ty alpha, Ty beta,
    size_t M, size_t N, size_t K, size_t batch_size
) {
    Accessor<Ty, A_layout> A_ = Accessor<Ty, A_layout>(A, M, K);
    Accessor<Ty, B_layout> B_ = Accessor<Ty, B_layout>(B, K, N);
    Accessor<Ty, C_layout> C_ = Accessor<Ty, C_layout>(C, M, N);
    Accessor<Ty, D_layout> D_ = Accessor<Ty, D_layout>(D, M, N);
    Accessor<Ty, col_major> bias_ = Accessor<Ty, col_major>(bias, M);
    for (size_t bid = 0; bid < batch_size; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            for (size_t cid = 0; cid < N; cid++) {
                Ty value = 0;
                for (size_t kid = 0; kid < K; kid++) {
                    value += A_(bid, rid, kid) * B_(bid, kid, cid);
                }
                value = alpha * value + beta * C_(bid, rid, cid) + bias_(bid, rid);
                D_(bid, rid, cid) = value > 0 ? value : 0;
            }
        }
    }
}