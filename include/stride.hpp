#pragma once

#include <cstddef>

namespace tensor_stride {
    /*
     * Calcula o stride row-major de um tensor contíguo (sem buracos na memória).
     * A última dimensão tem passo 1; cada uma das outras é o stride seguinte
     * multiplicado pelo tamanho da dimensão seguinte.
     *
     *   shape = [2, 3, 4]  ->  stride = [12, 4, 1]
     *
     * `stride` precisa ter espaço para `rank` posições.
     */
    void contiguous_stride(const size_t* shape, size_t rank, size_t* stride);
}
