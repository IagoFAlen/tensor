#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H
#pragma once

#include "domain/itensor.hpp"
#include "infrastructure/tensor.hpp"

namespace tensor_factory {
    typedef enum {
        TENSOR_CPU,
        TENSOR_GPU,
        TENSOR_AUTO
    } TensorType;

    bool has_gpu_support();
    ITensor* factory_create(TensorType type, size_t rank);
}

#endif