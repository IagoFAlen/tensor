#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "stride.hpp"

namespace HOST {
    typedef struct tensor {
        float*  data;
        size_t* shape;  // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride; // quantos passos preciso dar para pular de dimensão
        size_t  rank;   // rank é o tamanho do vector shape
        size_t  n;

        tensor(size_t* sh, size_t r): shape(sh), rank(r), n(1) {
            stride = (size_t*) malloc(rank * sizeof(size_t));
            tensor_stride::contiguous_stride(shape, rank, stride);

            for(size_t i = 0; i < rank; i++){
                n *= shape[i];
            }

            data = (float*) malloc(n * sizeof(float));
        }

        ~tensor(){
            n = 0;
            rank = 0;
            free(data);
        };

    } TENSOR;
}
