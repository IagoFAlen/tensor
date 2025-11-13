// tensor_factory.cpp
#include "tensor_factory.hpp"
#include "infrastructure/tensor_cpu.repository.hpp"
#include "infrastructure/tensor_gpu.repository.hpp"

namespace tensor_factory {
    bool has_gpu_support() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        return deviceCount > 0;
    }

    ITensor* factory_create(TensorType type, size_t rank, size_t* shape, float default_weight) {
        if (type == TENSOR_AUTO) {
            type = has_gpu_support() ? TENSOR_GPU : TENSOR_CPU;
        }

        if (type == TENSOR_GPU && has_gpu_support()) {
            return gpu_tensor::create_tensor(rank, shape, default_weight);
        } else {
            return cpu_tensor::create_tensor(rank, shape, default_weight);
        }
    }
}