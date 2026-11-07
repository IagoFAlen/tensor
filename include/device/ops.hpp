#pragma once

#include "device/tensor.hpp"
#include "status.hpp"

using namespace tensor_status;

// Operações de cálculo na GPU (kernels e as funções que os lançam).
namespace DEVICE {
    Status matmul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C);
}
