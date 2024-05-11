/**
 * All tensors stored in column-major format.
*/

#include <cuda.h>
#include <time.h>
#include <stdarg.h>

template<typename Ty>
Ty* memory_host(size_t num, Ty max_value = (Ty)(1)) {
    static time_t start_time = 0;
    if (start_time == 0) {
        srand(time(&start_time));
    }
    Ty* ptr = reinterpret_cast<Ty*>(malloc(sizeof(Ty) * num));
    for (size_t i = 0; i < num; i++) {
        ptr[i] = static_cast<Ty>(static_cast<float>(rand()) / RAND_MAX * max_value);
    }
    return ptr;
}

template<typename Ty>
Ty* memory_cuda(size_t num, Ty* host_ptr = nullptr) {
    Ty* ptr;
    cudaMalloc(&ptr, sizeof(Ty) * num);
    if (host_ptr != nullptr) {
        cudaMemcpy(ptr, host_ptr, sizeof(Ty) * num, cudaMemcpyHostToDevice);
    }
    return ptr;
}

void free_host(size_t length, ...) {
    va_list vlist;
    va_start(vlist, length);
    for (size_t i = 0; i < length; i++) {
        void *ptr = va_arg(vlist, void*);
        free(ptr);
    }
    va_end(vlist);
}

void free_cuda(size_t length, ...) {
    va_list vlist;
    va_start(vlist, length);
    for (size_t i = 0; i < length; i++) {
        void *ptr = va_arg(vlist, void*);
        cudaFree(ptr);
    }
    va_end(vlist);
}

template<typename Ty>
bool check_same(Ty *ptr1, Ty *ptr2, size_t num, Ty error = 1e-5) {
    for (size_t i = 0; i < num; i++) {
        if (abs(ptr1[i] - ptr2[i]) > error) return false;
    }
    return true;
}

template<typename Ty>
void gemv(
    int M, int N, const Ty alpha, const Ty beta,
    const Ty *A, const Ty *x, Ty *y
) {
    for (int r = 0; r < M; r++) {
        Ty value = 0;
        for (int c = 0; c < N; c++) {
            value += A[c * M + r] * x[c];
        }
        y[r] = alpha * value + beta * y[r];
    }
}