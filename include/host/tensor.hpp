#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "stride.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace HOST {
    typedef struct tensor {
        float*  data;
        size_t* shape;  // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride; // quantos passos preciso dar para pular de dimensão
        size_t  rank;   // rank é o tamanho do vector shape
        size_t  n;

        // O construtor só descreve o tensor; a memória vem do alloc(), que pode devolver Status.
        tensor(size_t* sh, size_t r): data(nullptr), shape(sh), stride(nullptr), rank(r), n(1) {
            for(size_t i = 0; i < rank; i++){
                n *= shape[i];
            }
        }

        Status alloc(){
            if(data != nullptr) return TENSOR_SUCCESS;     // já alocado

            stride = (size_t*) malloc(rank * sizeof(size_t));
            data   = (float*)  malloc(n * sizeof(float));

            if(stride == nullptr || data == nullptr){
                free(stride); stride = nullptr;
                free(data);   data   = nullptr;
                return TENSOR_HOST_ALLOC_FAILED;
            }

            tensor_stride::contiguous_stride(shape, rank, stride);
            return TENSOR_SUCCESS;
        }

        ~tensor(){
            n = 0;
            rank = 0;
            free(data);
            free(stride);
        };

        // O TENSOR é dono de data e de stride: uma cópia liberaria a mesma memória duas vezes (double free).
        tensor(const tensor&) = delete;
        tensor& operator=(const tensor&) = delete;

    } TENSOR;
}
