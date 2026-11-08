#pragma once

#include <cstddef>

// Configuração do lançamento dos kernels na GPU.
namespace DEVICE {
    /*
     * Tamanho do bloco nos kernels que percorrem uma matriz de C (mat_mul, hadamard_mul):
     * x -> coluna, y -> linha. 16 × 16 = 256 threads por bloco (o limite do CUDA é 1024).
     */
    constexpr unsigned BLOCK_X = 16;
    constexpr unsigned BLOCK_Y = 16;
}
