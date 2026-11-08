#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda_runtime.h>
#include "status.hpp"
#include "host/tensor.hpp"

using namespace tensor_status;

namespace DEVICE {
    typedef struct tensor {
        float* data;
        size_t* shape;      // o tamanho de cada dimensão, por exemplo [2, 3, 4] -> convenção usar do mais lento ao mais rápido
        size_t* stride;     // Quantos passos preciso dar para pular de dimensão
        size_t rank;        // rank é o tamanho do vector shape
        size_t n;
        
        // O construtor só descreve o tensor; a memória vem do alloc(), que pode devolver Status.
        tensor(size_t* sh, size_t* st, size_t r): data(nullptr), shape(sh), stride(st), rank(r), n(1) {
            for(size_t i = 0; i < rank; i++){
                n *= shape[i];
            }
        }

        Status alloc(){
            if(data != nullptr) return TENSOR_SUCCESS;     // já alocado

            if(cudaMalloc(&data, n * sizeof(float)) != cudaSuccess){
                data = nullptr;
                return TENSOR_ALLOC_FAILED;
            }

            return TENSOR_SUCCESS;
        }

        // Copia n floats de h_data para a GPU, sem conferir nada: o buffer precisa estar no layout do tensor.
        Status host_to_device(const float* h_data){
            if(data == nullptr){
                return TENSOR_HOST_TO_DEVICE_NOT_ALLOCATED;
            }

            if(h_data == nullptr){
                return TENSOR_HOST_TO_DEVICE_NULL_DATA;
            }

            return (cudaMemcpy(data, h_data, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) ? TENSOR_HOST_TO_DEVICE_COPY_FAILED : TENSOR_SUCCESS;

        }

        /*
         * Copia um HOST::TENSOR para a GPU, conferindo antes que os dois descrevem o mesmo tensor.
         * O host_to_device confia que o buffer tem n floats no layout certo; o feed confere:
         *   - mesmo rank e shape  -> mesma quantidade de elementos
         *   - mesmo stride        -> a cópia é byte a byte, então o layout precisa ser igual
         *                            (o HOST::TENSOR é sempre contíguo)
         */
        Status feed(const HOST::TENSOR* h){
            if(data == nullptr){
                return TENSOR_FEED_NOT_ALLOCATED;
            }

            if(h == nullptr || h->data == nullptr || h->stride == nullptr){
                return TENSOR_FEED_NULL_DATA;
            }

            if(h->rank != rank){
                return TENSOR_FEED_SHAPE_MISMATCH;
            }

            for(size_t i = 0; i < rank; i++){
                if(h->shape[i] != shape[i]) return TENSOR_FEED_SHAPE_MISMATCH;
            }

            for(size_t i = 0; i < rank; i++){
                if(h->stride[i] != stride[i]) return TENSOR_FEED_STRIDE_MISMATCH;
            }

            return host_to_device(h->data);
        }

        /*
         * Copia data da GPU para h_data, que precisa ter espaço para n floats.
         * Antes da cópia, espera os kernels terminarem: um erro dentro de um kernel
         * só aparece numa sincronização, e aí volta como TENSOR_EXEC_FAILED.
         */
        Status device_to_host(float* h_data) const {
            if(data == nullptr){
                return TENSOR_DEVICE_TO_HOST_NOT_ALLOCATED;
            }

            if(h_data == nullptr){
                return TENSOR_DEVICE_TO_HOST_NULL_DATA;
            }

            if(cudaDeviceSynchronize() != cudaSuccess){
                return TENSOR_EXEC_FAILED;
            }

            return (cudaMemcpy(h_data, data, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) ? TENSOR_DEVICE_TO_HOST_COPY_FAILED : TENSOR_SUCCESS;
        }
        
        ~tensor(){
            n = 0;
            rank = 0;
            cudaFree(data);
        };

        // O TENSOR é dono de data: uma cópia liberaria a mesma memória duas vezes (double free).
        // Para ver a mesma memória de outro jeito (ex.: transposta), use um TENSOR3.
        tensor(const tensor&) = delete;
        tensor& operator=(const tensor&) = delete;

    } TENSOR;

    // Tensor em 3 dimensões: [batch, linhas, colunas]
    // Não é dono de data (sem destrutor): copiar é seguro e serve de view do TENSOR.
    typedef struct tensor3 {
        float* data;
        size_t shape[3];
        size_t stride[3];
    } TENSOR3;

}
