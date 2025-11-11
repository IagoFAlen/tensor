#include "infrastructure/tensor_gpu.repository.hpp"
#include "domain/tensor_utils.hpp"

namespace gpu_tensor {
    __global__ void kernel_add_weight(float* matrix, size_t idx, float weight) {
        matrix[idx] = weight;
    }

    __global__ void kernel_get_weight(float* matrix, size_t idx, float* result) {
        *result = matrix[idx];
    }

    static void add_weights(void* self, float* weights){
        tensor::Tensor* tensor = (tensor::Tensor*) self;

        for(size_t i = 0; i < tensor->rank; ++i){
            tensor->weights[i] = weights[i];
        }
    }

    static void add_vertexes(void* self) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        cudaMalloc(&tensor->weights, tensor->weights_size * sizeof(float));
        cudaMemset(tensor->weights, 0, tensor->weights_size * sizeof(float));
    }

    static void add_weights_by_coord(void* self, size_t* coord, float weight) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        size_t idx = tensor->get_index(coord);

        kernel_add_weight<<<1, 1>>>(tensor->weights, idx, weight);
        cudaDeviceSynchronize();
    }

    static float get_weight_by_coord(void* self, size_t* coord) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        float result = 0.0f;
        float* d_result;

        size_t idx = tensor->get_index(coord);

        cudaMalloc(&d_result, sizeof(float));
        kernel_get_weight<<<1, 1>>>(tensor->weights, idx, d_result);
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);

        return result;
    }

    static void destroy(void* self) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        if (tensor->weights)
            cudaFree(tensor->weights);
        free(tensor);
    }

    ITensor* create_tensor(size_t rank, size_t* dimensions_size) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(malloc(sizeof(tensor::Tensor)));

        if(!tensor) return nullptr;

        tensor->interface->add_vertexes              = add_vertexes;
        tensor->interface->add_weights               = add_weights;
        tensor->interface->add_weights_by_coord      = add_weights_by_coord;
        tensor->interface->get_weight_by_coord       = get_weight_by_coord;
        tensor->interface->destroy                   = destroy;

        tensor->rank = rank;
        tensor->set_dimensions(dimensions_size);
        tensor->fill_stride();
        tensor->weights_size = tensor_utils::calc_weights_size(tensor->dimensions, tensor->rank);

        return reinterpret_cast<ITensor*>(tensor);
    }
}