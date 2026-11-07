#pragma once

#include "device/tensor.hpp"
#include "status.hpp"

using namespace tensor_status;

// Preparação dos tensores antes do cálculo (conversões de shape e stride, validações).
namespace DEVICE {
    Status to_tensor3();
}
