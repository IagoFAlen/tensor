#ifndef TENSOR_H
#define TENSOR_H
#pragma once

#include "domain/itensor.hpp"
#include <cuda_runtime.h>

using namespace std;

namespace tensor {
    typedef struct Tensor {
        ITensor* interface;
        float* weights;
        size_t* dimensions;
        size_t* stride;
        size_t rank;
        size_t weights_size;

        Tensor(): interface(nullptr), weights(nullptr),
                  dimensions(nullptr), stride(nullptr),
                  rank(0), weights_size(0) {}

        void fill_stride(){
            stride = (size_t*) malloc(rank * sizeof(size_t));
            stride[rank - 1] = 1;
            for (size_t i = rank - 1; i-- > 0;)
                stride[i] = stride[i + 1] * dimensions[i + 1];

        }

        size_t get_index(size_t* coords){
            size_t idx = 0;
            for(size_t i = 0; i < rank; ++i)
                idx += coords[i] * stride[i];

            return idx;
        }

        size_t* get_coords(size_t index) {
            size_t* coords = (size_t*) malloc(rank * sizeof(size_t));
            for (size_t i = 0; i < rank; ++i)
                coords[i] = (index / stride[i]) % dimensions[i];
            return coords;
        }

        void set_dimensions(const size_t* dimension_sizes) {
            if (dimensions)
                free(dimensions);

            dimensions = (size_t*) malloc(rank * sizeof(size_t));

            for (size_t i = 0; i < rank; ++i)
                dimensions[i] = dimension_sizes[i];
        }

    };

    ITensor* create_tensor(size_t rank);
}

#endif