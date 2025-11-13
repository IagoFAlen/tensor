#include "infrastructure/tensor_gpu.repository.hpp"
#include "domain/tensor_utils.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <algorithm>

const float ALIGMENT_BYTES_SIZE = 128;

namespace gpu_tensor {
    // Kernel para inicialização com valor
    __global__ void kernel_init_with_default_value(float* weights, size_t size, float value) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < size; i += stride) {
            weights[i] = value;
        }
    }

    // Kernel para incremento atômico
    __global__ void kernel_atomic_increment(unsigned long long* offset, unsigned long long increment, unsigned long long* result) {
        *result = atomicAdd(offset, increment);
    }

    __global__ void kernel_get_weight(float* matrix, size_t idx, float* result) {
        *result = matrix[idx];
    }

    // Função auxiliar para alinhar o bloco
    static float* align_block(float* weights_block, size_t block_size, size_t contiguous_size, size_t* padded_size, float default_weight) {
        size_t remainder = block_size % contiguous_size;
        if (remainder == 0) {
            *padded_size = block_size;
            return weights_block;
        }

        *padded_size = block_size + (contiguous_size - remainder);
        float* aligned_block;
        posix_memalign((void**)&aligned_block, ALIGMENT_BYTES_SIZE, *padded_size * sizeof(float));
        memcpy(aligned_block, weights_block, block_size * sizeof(float));
        std::fill(aligned_block + block_size, aligned_block + *padded_size, default_weight);
        return aligned_block;
    }

    static void add_block(void* self, float* weights_block, size_t block_size) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        size_t contiguous_size = tensor->stride[0];
        size_t padded_size;

        // Alinhar o bloco
        float* aligned_block = align_block(weights_block, block_size, contiguous_size, &padded_size, tensor->default_weight);

        // Criar um stream CUDA
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Obter offset atômico
        unsigned long long* d_result;
        cudaMalloc(&d_result, sizeof(unsigned long long));
        unsigned long long thread_offset;

        // Lançar kernel atômico no stream
        kernel_atomic_increment<<<1, 1, 0, stream>>>(tensor->offset, padded_size, d_result);

        // Copiar o resultado para a CPU de forma assíncrona
        cudaMemcpyAsync(&thread_offset, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);

        // Copiar bloco para a GPU de forma assíncrona
        cudaMemcpyAsync(&tensor->weights[thread_offset], aligned_block, padded_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Sincronizar o stream para garantir que todas as operações estejam completas
        cudaStreamSynchronize(stream);

        // Limpeza
        cudaFree(d_result);
        cudaStreamDestroy(stream);
        if (aligned_block != weights_block) {
            free(aligned_block);
        }
    }

    static void add_vertexes(void* self) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(self);
        cudaMalloc(&tensor->weights, tensor->weights_size * sizeof(float));
        cudaMalloc(&tensor->offset, sizeof(unsigned long long));
        cudaMemset(tensor->offset, 0, sizeof(unsigned long long));

        float value = tensor->default_weight; // NaN
        const int threads = 256;
        const int blocks = (tensor->weights_size + threads - 1) / threads;
        kernel_init_with_default_value<<<blocks, threads>>>(tensor->weights, tensor->weights_size, value);
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
        if (tensor->weights) {
            cudaFree(tensor->weights);
        }
        if (tensor->offset) {
            cudaFree(tensor->offset);
        }
        free(tensor);
    }

    ITensor* create_tensor(size_t rank, size_t* shape, float default_weight) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(malloc(sizeof(tensor::Tensor)));
        if (!tensor) return nullptr;

        tensor->interface = (ITensor*)malloc(sizeof(ITensor));
        if (!tensor->interface) {
            free(tensor);
            return nullptr;
        }

        tensor->interface->add_vertexes = add_vertexes;
        tensor->interface->add_block = add_block;
        tensor->interface->get_weight_by_coord = get_weight_by_coord;
        tensor->interface->destroy = destroy;

        tensor->rank = rank;
        tensor->set_shape(shape);
        tensor->fill_stride();
        tensor->weights_size = tensor_utils::calc_weights_size(tensor->shape, tensor->rank);
        tensor->default_weight = default_weight;

        tensor->interface->add_vertexes(tensor);

        return reinterpret_cast<ITensor*>(tensor);
    }
}