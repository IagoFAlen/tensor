#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda.h>
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
    typedef struct tensor {
        float* data;
        size_t* shape;      // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride;     // Quantos passos preciso dar para pular de dimensão
        size_t rank;        // rank é o tamanho do vector shape
        size_t n;
        
        tensor(size_t* sh, size_t* st, size_t r): shape(sh), stride(st), rank(r), n(1) {
            for(int i = 0; i < rank; i++){
                n *= shape[i];
            }

            cudaMalloc(&data, n * sizeof(float));
        }

        Status feed(const float* h_data){
            if(h_data == nullptr){
                return TENSOR_BAD_PARAM;
            }

            return (cudaMemcpy(data, h_data, n * sizeof(float), cudaMemcpyToDevice) != cudaSuccess) ? TENSOR_EXEC_FAILED : TENSOR_SUCCESS;
        
        }
        
        ~tensor(){
            n = 0;
            rank = 0;
            cudaFree(data);
        };

    } TENSOR;

    // Tensor em 3 dimensões: [batch, linhas, colunas]
    typedef struct tensor3 {
        float* data;
        size_t shape[3];
        size_t stride[3];
    } TENSOR3;

}
