#include "infrastructure/tensor_cpu.repository.hpp"
#include "domain/tensor_utils.hpp"
#include <limits>
#include <algorithm>

static pthread_mutex_t tensor_mutex = PTHREAD_MUTEX_INITIALIZER;
static float ALIGNMENT_SIZE_BYTES = 128;
namespace cpu_tensor {
    static void add_vertexes(void* self) {
        tensor::Tensor* tensor = (tensor::Tensor*)self;
        tensor->weights = (float*)malloc(tensor->weights_size * sizeof(float));
        tensor->offset = (unsigned long long*)malloc(sizeof(unsigned long long));
        *tensor->offset = 0;
        std::fill(tensor->weights, tensor->weights + tensor->weights_size, tensor->default_weight);
    }

    static void add_block(void* self, float* weights_block, size_t block_size) {
        tensor::Tensor* tensor = (tensor::Tensor*)self;

        pthread_mutex_lock(&tensor_mutex);

        size_t contiguous_size = tensor->stride[0];
        size_t remainder = block_size % contiguous_size;

        float* aligned_block = weights_block;
        size_t padded_size = block_size;
        if (remainder != 0) {
            padded_size = block_size + (contiguous_size - remainder);
            posix_memalign((void**)&aligned_block, ALIGNMENT_SIZE_BYTES, padded_size * sizeof(float));
            memcpy(aligned_block, weights_block, block_size * sizeof(float));
            std::fill(aligned_block + block_size, aligned_block + padded_size, tensor->default_weight);
        }

        memcpy(&tensor->weights[*tensor->offset], aligned_block, padded_size * sizeof(float));
        *tensor->offset += padded_size;

        if (aligned_block != weights_block) {
            free(aligned_block);
        }

        pthread_mutex_unlock(&tensor_mutex);
    }

    static float get_weight_by_coord(void* self, size_t* coord){
        tensor::Tensor* tensor = (tensor::Tensor*) self;
        
        size_t idx = tensor->get_index(coord);
        
        if(tensor_utils::valid_index(idx, tensor->weights_size))
            return tensor->weights[idx];
    }

    static void destroy(void* self) {
        tensor::Tensor* tensor = (tensor::Tensor*)self;
        if (tensor->weights) {
            free(tensor->weights);
        }
        if (tensor->offset) {
            free(tensor->offset);
        }
        free(tensor);
    }

    ITensor* create_tensor(size_t rank, size_t* shape, float default_weight) {
        tensor::Tensor* tensor = (tensor::Tensor*)malloc(sizeof(tensor::Tensor));
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