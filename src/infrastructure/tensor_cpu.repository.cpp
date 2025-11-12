#include "infrastructure/tensor_cpu.repository.hpp"
#include "domain/tensor_utils.hpp"

namespace cpu_tensor {
    static void add_vertexes(void* self){
        tensor::Tensor* tensor = (tensor::Tensor*) self;
        tensor->weights = (float*)malloc(tensor->weights_size * sizeof(float));
        memset(tensor->weights, 0, tensor->weights_size * sizeof(float));
    }

    static void add_weights(void* self, float* weights){
        tensor::Tensor* tensor = (tensor::Tensor*) self;

        for(size_t i = 0; i < tensor->weights_size; ++i){
            tensor->weights[i] = weights[i];
        }
    }

    static size_t* add_block(void* self, size_t* start_coord, size_t* block_shape, float* weights_block) {
        tensor::Tensor* tensor = (tensor::Tensor*) self;
        size_t start_idx = 0;
        size_t offset = 0;

        start_idx = tensor->get_index(start_coord);
        size_t block_size = 1;
        
        for(size_t i = 0; i < tensor->rank; ++i){
            block_size *= block_shape[i];
        }
        
        offset = start_idx;
        
        for(size_t i = start_idx; i < start_idx + block_size; ++i){
            memcpy(&tensor->weights[start_idx], &weights_block, block_size * sizeof(float));
        }

        return tensor->get_coords(offset);

    }

    static void add_weights_by_coord(void* self, size_t* coord, float weight){
        tensor::Tensor* tensor = (tensor::Tensor*) self;

        size_t idx = tensor->get_index(coord);

        if(tensor_utils::valid_index(idx, tensor->weights_size))
            tensor->weights[idx] = weight;
    }

    static float get_weight_by_coord(void* self, size_t* coord){
        tensor::Tensor* tensor = (tensor::Tensor*) self;
        
        size_t idx = tensor->get_index(coord);
        
        if(tensor_utils::valid_index(idx, tensor->weights_size))
            return tensor->weights[idx];
    }

    static void destroy(void* self){
        tensor::Tensor* g = static_cast<tensor::Tensor*>(self);
        free(g);
    }

    ITensor* create_tensor(size_t rank, size_t* shape) {
        tensor::Tensor* tensor = static_cast<tensor::Tensor*>(malloc(sizeof(tensor::Tensor)));

        if(!tensor) return nullptr;

        tensor->interface = (ITensor*) malloc(sizeof(ITensor));

        tensor->interface->add_vertexes              = add_vertexes;
        tensor->interface->add_weights               = add_weights;
        tensor->interface->add_weights_by_coord      = add_weights;
        tensor->interface->get_weight_by_coord       = get_weight;
        tensor->interface->destroy                   = destroy;

        tensor->rank = rank;
        tensor->set_shape(shape);
        tensor->fill_stride();
        tensor->weights_size = tensor_utils::calc_weights_size(tensor->shape, tensor->rank);

        return reinterpret_cast<ITensor*>(tensor);
    }
}