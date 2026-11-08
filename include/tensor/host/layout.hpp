#pragma once

#include "tensor/host/tensor.hpp"
#include "tensor/status.hpp"

using namespace tensor_status;

// Preparação dos tensores antes do cálculo (conversões de shape e stride, validações).
namespace HOST {
    Status validate_matmul(const TENSOR* A, const TENSOR* B);
    Status validate_operations(const TENSOR* A, const TENSOR* B);
}
