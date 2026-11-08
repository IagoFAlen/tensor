#pragma once

#include "tensor/device/tensor.hpp"
#include "tensor/status.hpp"

using namespace tensor_status;

// Operações de cálculo na GPU (kernels e as funções que os lançam).
namespace DEVICE {
    Status mat_mul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C);
    Status hadamard_mul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C);

    // Espera os kernels lançados terminarem. TENSOR_EXEC_FAILED se algum falhou durante a execução.
    Status synchronize();
}
