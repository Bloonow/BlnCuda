/**
 * All vectors and matrices stored in column-major format.
 * 🤣 👉 🤡
*/

#include <cuda.h>
#include <time.h>
#include <stdarg.h>

template<typename Ty>
Ty* allocHostMemory(size_t count, Ty max_init = (Ty)(1)) {
    static time_t init_time = 0;
    if (init_time == 0) srand(time(&init_time));
    Ty* ptr = reinterpret_cast<Ty*>(malloc(sizeof(Ty) * count));
    for (size_t i = 0; i < count; i++) {
        ptr[i] = static_cast<Ty>(static_cast<double>(rand()) / RAND_MAX * max_init);
    }
    return ptr;
}

template<typename Ty>
Ty* allocCudaMemory(size_t count, Ty *host_ptr = nullptr) {
    Ty* ptr;
    cudaMalloc(&ptr, sizeof(Ty) * count);
    if (host_ptr != nullptr) {
        cudaMemcpy(ptr, host_ptr, sizeof(Ty) * count, cudaMemcpyHostToDevice);
    }
    return ptr;
}

void freeMemory(size_t count, ...) {
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
bool checkSame(Ty *ptr1, Ty *ptr2, size_t num, Ty error = 1e-4) {
    for (size_t i = 0; i < num; i++) {
        if (abs(ptr1[i] - ptr2[i]) > error) return false;
    }
    return true;
}

template<typename Ty>
void host_gemv(
    const size_t M, const size_t N, const Ty alpha, const Ty beta,
    const Ty *A, const Ty *x, Ty *y, const size_t batchCount
) {
    for (size_t bid = 0; bid < batchCount; bid++) {
        for (size_t rid = 0; rid < M; rid++) {
            Ty value = 0;
            for (size_t cid = 0; cid < N; cid++) {
                value += A[bid * M * N + cid * M + rid] * x[bid * N + cid];
            }
            y[bid * M + rid] = alpha * value + beta * y[bid * M + rid];
        }
    }
}
