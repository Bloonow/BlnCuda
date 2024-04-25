#pragma once

#define time_it(__call, __repeat) do {                        \
    cudaEvent_t __start, __end;                               \
    cudaEventCreate(&__start);                                \
    cudaEventCreate(&__end);                                  \
    cudaEventRecord(__start);                                 \
    cudaEventQuery(__start);                                  \
    (__call);                                                 \
    for (int __i = 0; __i < __repeat; __i++) (__call);        \
    cudaEventRecord(__end);                                   \
    cudaEventSynchronize(__end);                              \
    float __elapse;                                           \
    cudaEventElapsedTime(&__elapse, __start, __end);          \
    cudaEventDestroy(__start);                                \
    cudaEventDestroy(__end);                                  \
    printf("Elapse = %g ms, Average = %g ms, Repeat = %d.\n", \
            __elapse, __elapse / __repeat, __repeat);         \
} while (0);
// #define time_it(__call, __repeat)

#define time_it_v2(__call, __repeat, __ave) do {              \
    cudaEvent_t __start, __end;                               \
    cudaEventCreate(&__start);                                \
    cudaEventCreate(&__end);                                  \
    cudaEventRecord(__start);                                 \
    cudaEventQuery(__start);                                  \
    (__call);                                                 \
    for (int __i = 0; __i < __repeat; __i++) (__call);        \
    cudaEventRecord(__end);                                   \
    cudaEventSynchronize(__end);                              \
    float __elapse;                                           \
    cudaEventElapsedTime(&__elapse, __start, __end);          \
    __ave = __elapse / __repeat;                              \
    cudaEventDestroy(__start);                                \
    cudaEventDestroy(__end);                                  \
} while (0);
// #define time_it_v2(__call, __repeat, __ave)