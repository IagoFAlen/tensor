#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "tensor/stride.hpp"
#include "tensor/status.hpp"

using namespace tensor_status;

namespace HOST {
    typedef struct tensor {
        float*  data;
        size_t* shape;  // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride; // quantos passos preciso dar para pular de dimensão
        size_t  rank;   // rank é o tamanho do vector shape
        size_t  n;

        /*
         * O construtor só guarda o rank; toda memória vem dos alloc_*, que devolvem Status:
         *
         *     HOST::TENSOR h(3);
         *     h.alloc_shape(shape);     // copia shape (rank posições) e calcula n
         *     h.alloc_data();           // n floats na RAM, e o stride contíguo calculado do shape
         *
         * shape é COPIADO: o tensor não depende de o array de quem chamou continuar existindo.
         */
        explicit tensor(size_t r): data(nullptr), shape(nullptr), stride(nullptr), rank(r), n(1) {}

        // malloc(0) pode devolver nullptr, o que pareceria falha: aloca pelo menos 1 posição.
        size_t dims_bytes() const { return (rank > 0 ? rank : 1) * sizeof(size_t); }

        Status alloc_shape(const size_t* sh){
            if(sh == nullptr)     return TENSOR_SHAPE_NULL_PARAM;
            if(shape != nullptr)  return TENSOR_SHAPE_ALREADY_ALLOCATED;

            shape = (size_t*) malloc(dims_bytes());
            if(shape == nullptr)  return TENSOR_SHAPE_ALLOC_FAILED;

            n = 1;
            for(size_t i = 0; i < rank; i++){
                shape[i] = sh[i];
                n *= sh[i];
            }

            return TENSOR_SUCCESS;
        }

        // Precisa do shape antes: é dele que saem n e o stride contíguo.
        Status alloc_data(){
            if(shape == nullptr) return TENSOR_DATA_SHAPE_NOT_ALLOCATED;
            if(data != nullptr) return TENSOR_SUCCESS;     // já alocado

            stride = (size_t*) malloc(dims_bytes());
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
            free(shape);
            free(stride);
        };

        // O TENSOR é dono de data, shape e stride: uma cópia liberaria a mesma memória duas vezes (double free).
        tensor(const tensor&) = delete;
        tensor& operator=(const tensor&) = delete;

    } TENSOR;
}
