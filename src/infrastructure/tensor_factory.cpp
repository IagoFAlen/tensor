#include "infrastructure/tensor_factory.hpp"
#include "infrastructure/tensor_cpu.repository.hpp"
#include "infrastructure/tensor_gpu.repository.hpp"

namespace tensor_factory {
    static bool has_gpu_support() {
        return false;
    }

    ITensor* factory_create(TensorType type, size_t rank, size_t* dimensions_size) {
        switch (type) {
            case TENSOR_CPU:
                return cpu_tensor::create_tensor(rank, dimensions_size);
            case TENSOR_GPU:
                return gpu_tensor::create_tensor(rank, dimensions_size);
            case TENSOR_AUTO:
                if (!has_gpu_support())
                    return cpu_tensor::create_tensor(rank, dimensions_size);
                else
                    return gpu_tensor::create_tensor(rank, dimensions_size);
            default:
                return nullptr;
        }
    }
}
