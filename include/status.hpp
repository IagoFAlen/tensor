#pragma once

#include <cstddef>

namespace tensor_status {
    /*
     * Um status por verificação: o nome do status já diz onde a execução falhou.
     */
    typedef enum Status {
        TENSOR_SUCCESS,

        // Genéricos
        TENSOR_BAD_PARAM,
        TENSOR_NOT_SUPPORTED,
        TENSOR_EXEC_FAILED,             // synchronize, fetch: um kernel lançado antes falhou durante a execução

        // Tensor (device/tensor.hpp)
        TENSOR_ALLOC_FAILED,            // alloc: cudaMalloc falhou
        TENSOR_FEED_NOT_ALLOCATED,      // feed: data == nullptr (alloc não foi chamado ou falhou)
        TENSOR_FEED_NULL_DATA,          // feed: h_data == nullptr
        TENSOR_COPY_FAILED,             // feed: cudaMemcpy falhou
        TENSOR_FETCH_NOT_ALLOCATED,     // fetch: data == nullptr (alloc não foi chamado ou falhou)
        TENSOR_FETCH_NULL_DATA,         // fetch: h_data == nullptr
        TENSOR_FETCH_COPY_FAILED,       // fetch: cudaMemcpy falhou

        // Tensor (host/tensor.hpp)
        TENSOR_HOST_ALLOC_FAILED,       // alloc: malloc de data ou de stride falhou

        // Layout (device/layout.cu)
        TENSOR_LAYOUT_NULL_PARAM,       // to_tensor3: in, out, in->shape ou in->stride == nullptr
        TENSOR_RANK_TOO_SMALL,          // to_shape3, to_stride3, validate_matmul: rank < 2
        TENSOR_BATCH_NOT_CONTIGUOUS,    // to_tensor3: as dimensões de batch não podem ser fundidas
        TENSOR_TRANSPOSE_NULL_PARAM,    // transpose: in ou out == nullptr

        // Matmul (host/layout.cpp, device/ops.cu)
        TENSOR_MATMUL_NULL_PARAM,       // validate_matmul, mat_mul: A, B, C ou algum data/shape == nullptr
        TENSOR_MATMUL_K_MISMATCH,       // validate_matmul, mat_mul: colunas de A != linhas de B
        TENSOR_MATMUL_BATCH_MISMATCH,   // mat_mul: batch de A ou de B diferente do de C, sem broadcast (shape[0] == 1 e stride[0] == 0)
        TENSOR_MATMUL_C_SHAPE,          // mat_mul: linhas de C != linhas de A, ou colunas de C != colunas de B

        // Hadamard (device/ops.cu)
        TENSOR_HADAMARD_NULL_PARAM,     // hadamard_mul: A, B, C ou algum data == nullptr
        TENSOR_HADAMARD_SHAPE_MISMATCH, // hadamard_mul: linhas ou colunas de A ou de B diferentes das de C
        TENSOR_HADAMARD_BATCH_MISMATCH, // hadamard_mul: batch de A ou de B diferente do de C, sem broadcast (shape[0] == 1 e stride[0] == 0)

        // Lançamento (device/ops.cu)
        TENSOR_LAUNCH_FAILED,           // mat_mul, hadamard_mul: o lançamento do kernel falhou

        TENSOR_STATUS_COUNT
    } Status;

    const char* status_name(Status s);
}
