#pragma once

#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <cuda.h>

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
    Ty& operator()(size_t bid, size_t rid, size_t cid) {
        return data[bid * rows * cols + rid * cols + cid];
    }
};
template<typename Ty>
struct Accessor<Ty, col_major> {
    Ty* data;
    size_t rows, cols;
    Accessor(Ty* data_, size_t rows_, size_t cols_) : data(data_), rows(rows_), cols(cols_) {}
    Ty& operator()(size_t bid, size_t rid, size_t cid) {
        return data[bid * rows * cols + cid * rows + rid];
    }
};

template<typename Ty>
struct VectorAccessor {
    Ty* data;
    size_t stride;
    VectorAccessor(Ty* data_, size_t stride_) : data(data_), stride(stride_) {}
    Ty& operator()(size_t bid, size_t idx) {
        return data[bid * stride + idx];
    }
};


template<typename A_type, typename A_layout, typename acc_type = A_type>
void host_gemv(
    A_type* A, acc_type* x, acc_type* y, acc_type alpha, acc_type beta,
    size_t M, size_t N, size_t batch_size
) {
    Accessor<A_type, A_layout> A_ = Accessor<A_type, A_layout>(A, M, N);
    VectorAccessor<acc_type> x_ = VectorAccessor<acc_type>(x, N);
    VectorAccessor<acc_type> y_ = VectorAccessor<acc_type>(y, M);
    for (size_t bid = 0; bid < batch_size; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            acc_type value = 0;
            for (size_t kid = 0; kid < N; kid++) {
                value += static_cast<acc_type>(A_(bid, rid, kid)) * x_(bid, kid);
            }
            y_(bid, rid) = alpha * value + beta * y_(bid, rid);
        }
    }
}

template<typename A_type, typename A_layout, typename B_type, typename B_layout, typename C_type, typename C_layout, 
    typename D_type = C_type, typename D_layout = C_layout, typename acc_type = C_type>
void host_gemm(
    A_type* A, B_type* B, C_type* C, D_type* D, acc_type alpha, acc_type beta,
    size_t M, size_t N, size_t K, size_t batch_size
) {
    Accessor<A_type, A_layout> A_ = Accessor<A_type, A_layout>(A, M, K);
    Accessor<B_type, B_layout> B_ = Accessor<B_type, B_layout>(B, K, N);
    Accessor<C_type, C_layout> C_ = Accessor<C_type, C_layout>(C, M, N);
    Accessor<D_type, D_layout> D_ = Accessor<D_type, D_layout>(D, M, N);
    if (C != nullptr && beta != 0) {
        for (size_t bid = 0; bid < batch_size; bid++) {
            for (size_t rid = 0; rid < M; rid++) {
                for (size_t cid = 0; cid < N; cid++) {
                    acc_type value = 0;
                    for (size_t kid = 0; kid < K; kid++) {
                        value += static_cast<acc_type>(A_(bid, rid, kid)) * static_cast<acc_type>(B_(bid, kid, cid));
                    }
                    D_(bid, rid, cid) = alpha * value + beta * static_cast<acc_type>(C_(bid, rid, cid));
                }
            }
        }
    } else {
        for (size_t bid = 0; bid < batch_size; bid++) {
            for (size_t rid = 0; rid < M; rid++) {
                for (size_t cid = 0; cid < N; cid++) {
                    acc_type value = 0;
                    for (size_t kid = 0; kid < K; kid++) {
                        value += static_cast<acc_type>(A_(bid, rid, kid)) * static_cast<acc_type>(B_(bid, kid, cid));
                    }
                    D_(bid, rid, cid) = alpha * value;
                }
            }
        }
    }
}

template<typename A_type, typename A_layout, typename B_type, typename B_layout, typename C_type, typename C_layout, 
    typename D_type = C_type, typename D_layout = C_layout, typename acc_type = C_type>
void host_matmul_relu(
    A_type* A, B_type* B, C_type* C, D_type* D, acc_type* bias, acc_type alpha, acc_type beta,
    size_t M, size_t N, size_t K, size_t batch_size
) {
    Accessor<A_type, A_layout> A_ = Accessor<A_type, A_layout>(A, M, K);
    Accessor<B_type, B_layout> B_ = Accessor<B_type, B_layout>(B, K, N);
    Accessor<C_type, C_layout> C_ = Accessor<C_type, C_layout>(C, M, N);
    Accessor<D_type, D_layout> D_ = Accessor<D_type, D_layout>(D, M, N);
    VectorAccessor<acc_type> bias_ = VectorAccessor<acc_type>(bias, M);
    if (C != nullptr && beta != 0) {
        for (size_t bid = 0; bid < batch_size; bid++) {
            for (size_t rid = 0; rid < M; rid++) {
                for (size_t cid = 0; cid < N; cid++) {
                    acc_type value = 0;
                    for (size_t kid = 0; kid < K; kid++) {
                        value += static_cast<acc_type>(A_(bid, rid, kid)) * static_cast<acc_type>(B_(bid, kid, cid));
                    }
                    value = alpha * value + beta * static_cast<acc_type>(C_(bid, rid, cid)) + bias_(bid, rid);
                    D_(bid, rid, cid) = value > 0 ? value : 0;
                }
            }
        }
    } else {
        for (size_t bid = 0; bid < batch_size; bid++) {
            for (size_t rid = 0; rid < M; rid++) {
                for (size_t cid = 0; cid < N; cid++) {
                    acc_type value = 0;
                    for (size_t kid = 0; kid < K; kid++) {
                        value += static_cast<acc_type>(A_(bid, rid, kid)) * static_cast<acc_type>(B_(bid, kid, cid));
                    }
                    value = alpha * value + bias_(bid, rid);
                    D_(bid, rid, cid) = value > 0 ? value : 0;
                }
            }
        }
    }
}
