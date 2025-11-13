#ifndef TENSOR_H
#define TENSOR_H
#pragma once

#include "domain/itensor.hpp"
#include <cuda_runtime.h>
#include <limits>

using namespace std;

namespace tensor {
    typedef struct Tensor {
        ITensor* interface;     // Interface genérica do Tensor
        float* weights;         // Pesos do Tensor
        size_t* shape;          // Tamanho de cada uma das dimensões N-rank
        size_t* stride;         // Passos necessários para mudar de dimensão
        size_t rank;            // Quantidade de dimensões
        size_t weights_size;    // Quantidade total de pesos no Tensor
        unsigned long long* offset; // Contador para o último bloco (GPU: device, CPU: host)
        float default_weight;    // Valor padrão para inicialização de pesos

        Tensor() : interface(nullptr), weights(nullptr),
                   shape(nullptr), stride(nullptr),
                   rank(0), weights_size(0), offset(nullptr),
                   default_weight(std::numeric_limits<float>::quiet_NaN()) {}

        void fill_stride() {
            stride = (size_t*)malloc(rank * sizeof(size_t));
            stride[rank - 1] = 1;
            for (size_t i = rank - 1; i-- > 0;)
                stride[i] = stride[i + 1] * shape[i + 1];
        }

        size_t get_index(size_t* coords) {
            size_t idx = 0;
            for (size_t i = 0; i < rank; ++i)
                idx += coords[i] * stride[i];
            return idx;
        }

        size_t* get_coords(size_t index) {
            size_t* coords = (size_t*)malloc(rank * sizeof(size_t));
            for (size_t i = 0; i < rank; ++i)
                coords[i] = (index / stride[i]) % shape[i];
            return coords;
        }

        void set_shape(const size_t* dimension_sizes) {
            if (shape)
                free(shape);
            shape = (size_t*)malloc(rank * sizeof(size_t));
            for (size_t i = 0; i < rank; ++i)
                shape[i] = dimension_sizes[i];
        }
    };

    ITensor* create_tensor(size_t rank, size_t* shape, float default_weight = std::numeric_limits<float>::quiet_NaN());
}

#endif