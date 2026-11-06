#include "d_tensor.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
    __global__ void mul_kernel(){

    }

    __global__ void matmul_kernel(float* a_data, float* b_data, float* c_data, size_t M, size_t K, size_t N){
        
        size_t row      = blockIdx.y * blockDim.y + threadIdx.y;
        size_t column   = blockIdx.x * blockDim.x + threadIdx.x;
        size_t depth    = blockIdx.z * blockDim.z + threadIdx.z;

        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += a_data[(row * K) + k] * b_data[(N * k) + column];   // A[row][k] · B[k][column]
        }
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
        
        /*
         * Dimensões da multiplicação C = A · B, lidas das duas últimas dimensões
         * de cada tensor (row-major: shape[rank - 2] = linhas, shape[rank - 1] = colunas).
         *
         *   A: M × K      B: K × N      C: M × N
         *
         *   M -> linhas de A, que também são as linhas de C.
         *   K -> colunas de A, que devem ser iguais às linhas de B. É a dimensão
         *        "interna", consumida pela soma: C[row][col] = Σ_k A[row][k] · B[k][col].
         *        Por isso K não aparece em C.
         *   N -> colunas de B, que também são as colunas de C.
         *
         * Cada thread calcula um elemento C[row][col], com row em [0, M) e col em [0, N);
         * o laço em k percorre [0, K).
         */
        size_t M = A->shape[A->rank - 2];
        size_t K = A->shape[A->rank - 1];
        size_t N = B->shape[B->rank - 1];

        matmul_kernel<<<blocks, threads>>>(A->data, B->data, C->data, M, N, K);

        return cudaGetLastError() == cudaSuccess ? TENSOR_OK : TENSOR_EXEC_FAILED;
    }

}
