#include "device/layout.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
        /*
         * Funde o batch numa só dimensão; linha e coluna vão para as posições 1 e 2.
         *   rank 2: [3, 4]          -> [1, 3, 4]
         *   rank 3: [2, 3, 4]       -> [2, 3, 4]
         *   rank 4: [2, 4, 3, 4]    -> [2*4, 3, 4]   = [8, 3, 4]
         *   rank 5: [2, 2, 4, 3, 4] -> [2*2*4, 3, 4] = [16, 3, 4]
         */
        static Status to_shape3(const size_t* shape, size_t rank, size_t shape3[3]){
            if(rank < 2) return TENSOR_RANK_TOO_SMALL;
            size_t depth = 1;

            for(size_t i = 0; i < rank - 2; i++){
                depth *= shape[i];
            }

            shape3[0] = depth;
            shape3[1] = shape[rank - 2];
            shape3[2] = shape[rank - 1];

            return TENSOR_SUCCESS;
        }

        /*
         * st3 = [st[r-3], st[r-2], st[r-1]]  (passo do batch, da linha e da coluna)
         *   rank 2: [4, 1]             -> [0, 4, 1]   (sem batch)
         *   rank 3: [12, 4, 1]         -> [12, 4, 1]
         *   rank 4: [48, 12, 4, 1]     -> [12, 4, 1]  (48 sai)
         *   rank 5: [96, 48, 12, 4, 1] -> [12, 4, 1]  (96 e 48 saem)
         *
         * Os passos que saem eram de dimensões de batch fundidas no shape3[0]: no tensor
         * 3D elas não existem mais. Com o batch contíguo, st[r-3] já percorre o batch
         * inteiro (rank 4: 48 = 12 * 4 = st[1] * shape[1]).
         */
        static Status to_stride3(const size_t* stride, size_t rank, size_t stride3[3]){
            if (rank < 2) return TENSOR_RANK_TOO_SMALL;
            
            stride3[0] = (rank == 2) ? 0 : stride[rank - 3];
            stride3[1] = stride[rank - 2];  
            stride3[2] = stride[rank - 1];
            
            return TENSOR_SUCCESS;
        }

        static Status check_contiguous(const TENSOR* in){
            for(size_t i = 0; i + 3 < in->rank; i++){
                if(in->stride[i] != in->stride[i + 1] * in->shape[i + 1]) 
                    return TENSOR_BATCH_NOT_CONTIGUOUS;
            }

            return TENSOR_SUCCESS;
        }

        Status to_tensor3(const TENSOR* in, TENSOR3* out){
            if(in == nullptr || out == nullptr || in->shape == nullptr || in->stride == nullptr)
                return TENSOR_LAYOUT_NULL_PARAM;

            Status st;
            // check contiguous
            st = check_contiguous(in);
            if(st != TENSOR_SUCCESS) return st;
            // convert +3D-rank shape into 3D
            st = to_shape3(in->shape, in->rank, out->shape);
            if(st != TENSOR_SUCCESS) return st;

            // copy stride to stride 3D
            st = to_stride3(in->stride, in->rank, out->stride);
            if(st != TENSOR_SUCCESS) return st;
        
            out->data = in->data;

            return TENSOR_SUCCESS;
        }

        /*
         * [batch, M, N] -> [batch, N, M]: mesma memória, linha e coluna trocadas.
         * Não copia dados: só troca shape e stride das posições 1 e 2; o kernel lê pelos strides.
         *   shape [2, 3, 4], stride [12, 4, 1] -> shape [2, 4, 3], stride [12, 1, 4]
         * in == out é permitido (transpõe no lugar).
         */
        Status transpose(const TENSOR3* in, TENSOR3* out){
            if(in == nullptr || out == nullptr) return TENSOR_TRANSPOSE_NULL_PARAM;

            TENSOR3 t = *in;        // cópia local: funciona mesmo se in == out
            t.shape[1]  = in->shape[2];
            t.shape[2]  = in->shape[1];
            t.stride[1] = in->stride[2];
            t.stride[2] = in->stride[1];
            *out = t;

            return TENSOR_SUCCESS;
        }
}
