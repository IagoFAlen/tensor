/*
 * Emulação do CUDA na CPU, só para testar sem GPU (build com -DTENSOR_EMULATE=ON).
 *
 * - A memória "da GPU" é memória comum (malloc), e cudaMemcpy é memcpy.
 * - Um kernel roda percorrendo o grid inteiro, thread por thread, em sequência.
 *   O CMake troca  kernel<<<grid, block>>>(args)  por  LAUNCH(kernel, grid, block, args).
 *
 * Não emula paralelismo, shared memory nem __syncthreads(): serve para conferir índices,
 * strides, bordas do grid e os Status, não desempenho nem condições de corrida.
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 1 } cudaError_t;
typedef enum { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault } cudaMemcpyKind;

// Permite aos testes simular um erro dentro de um kernel: a próxima sincronização devolve este erro.
inline cudaError_t& emulate_pending_kernel_error(){ static cudaError_t e = cudaSuccess; return e; }

template <typename T> inline cudaError_t cudaMalloc(T** p, size_t n){ *p = (T*)malloc(n); return *p ? cudaSuccess : cudaErrorUnknown; }
inline cudaError_t cudaFree(void* p){ free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind){ memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaGetLastError(){ return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize(){
    cudaError_t e = emulate_pending_kernel_error();
    emulate_pending_kernel_error() = cudaSuccess;
    return e;
}

#define __global__

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
static dim3 blockIdx, threadIdx, blockDim, gridDim;

#define LAUNCH(kernel, grid, block, ...) do {                                         \
    dim3 g_ = dim3(grid), b_ = dim3(block); gridDim = g_; blockDim = b_;              \
    for (blockIdx.z = 0; blockIdx.z < g_.z; blockIdx.z++)                             \
    for (blockIdx.y = 0; blockIdx.y < g_.y; blockIdx.y++)                             \
    for (blockIdx.x = 0; blockIdx.x < g_.x; blockIdx.x++)                             \
    for (threadIdx.z = 0; threadIdx.z < b_.z; threadIdx.z++)                          \
    for (threadIdx.y = 0; threadIdx.y < b_.y; threadIdx.y++)                          \
    for (threadIdx.x = 0; threadIdx.x < b_.x; threadIdx.x++)                          \
        kernel(__VA_ARGS__);                                                          \
} while (0)
