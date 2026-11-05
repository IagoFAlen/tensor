#include "d_tensor.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
    __global__ void mul_kernel(){

    }

    __global__ void matmul_kernel(float* a_data, float* b_data, float* c_data, size_t a_rank, size_t b_rank, size_t c_rank, size_t* a_shape, size_t* b_shape, size_t* c_shape, size_t* a_stride, size_t* b_stride, size_t* c_stride){
        
        size_t row      = blockIdx.y * blockDim.y + threadIdx.y;
        size_t column   = blockIdx.x * blockDim.x + threadIdx.x;
        size_t depth    = blockIdx.z * blockDim.z + threadIdx.z;

        sum += a_data[(row * a_shape[a_rank - 1]) + column] * b_data[(row * b_shape[b_rank - 1]) + column]
        c_data[i] = sum;
    }
    
    ROW * SHAPE + COLUMN;
    A = 	| 1 2 3 | 
            | 4 5 6 |
            | 7 8 9 |
    
    B =     | 1 |
            | 2 |
            | 3 |

    ROW * SHAPE + COLUMN    
    0 * 3 + 0 = 0
    0 * 3 + 1 = 1
    0 * 3 + 2 = 2
    1 * 3 + 0 = 3
    1 * 3 + 1 = 4
    1 * 3 + 2 = 5
    2 * 3 + 0 = 6

    Status matmul(const TENSOR* A, const TENSOR* B, TENSOR* C){
        size_t threads = 256;
        size_t blocks  = (C->n * threads - 1) / threads; // to-do
        
        matmul_kernel<<<blocks, threads>>>(
            A->data,
            B->data,
            C->data,
            A->rank,
            B->rank,
            C->rank,
            A->shape,
            B->shape,
            C->shape,
            A->stride,
            B->stride,
            C->stride,
        );

        return cudaGetLastError() == cudaSuccess ? TENSOR_OK : TENSOR_EXEC_FAILED;
    }

}
