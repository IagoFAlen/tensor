#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda.h>

namespace DEVICE {
    typedef struct tensor {
        float* data;
        size_t* shape;      // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride;     // Quantos passos preciso dar para pular de dimensão
        size_t rank;        // rank é o tamanho do vector shape
        size_t n;
        
        tensor(float* h_data, size_t* sh, size_t* st, size_t r): shape(sh), stride(st), rank(r) {
            for(int i = 0; i < rank; i++){
                n *= shape[i];
            }

            cudaMalloc(&data, n * sizeof(float));
            cudaMemCpy(data, h_data, n * sizeof(float), cudaMemCpyToDevice);
        }

        ~tensor(){
            n = 0;
            rank = 0;
            cudaFree(data);
        };

    } TENSOR;

    bool validate_matrix_operation(TENSOR* A, TENSOR* B){
        
    }

}
