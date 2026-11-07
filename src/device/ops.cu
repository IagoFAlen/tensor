#include "device/ops.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
    __global__ void mul_kernel(){

    }

    __global__ void matmul_kernel(TENSOR3 A, TENSOR3 B, TENSOR3 C, size_t M, size_t K, size_t N){

        size_t row      = blockIdx.y * blockDim.y + threadIdx.y;
        size_t column   = blockIdx.x * blockDim.x + threadIdx.x;
        size_t depth    = blockIdx.z * blockDim.z + threadIdx.z;

        // O grid é arredondado para cima (número inteiro de blocos), então os blocos
        // da borda podem ter threads com depth >= batch, row >= M ou column >= N.
        // Elas não correspondem a nenhum elemento de C e precisam sair antes de
        // acessar a memória.
        if (depth >= C.shape[0] || row >= M || column >= N) return;
        /*
            * Referência: a mesma soma escrita com M, K e N, válida quando os tensores
            * são contíguos (row-major), ou seja, quando os strides valem:
            *
            *   A: shape [batch, M, K] -> stride [M * K, K, 1]
            *   B: shape [batch, K, N] -> stride [K * N, N, 1]
            *   C: shape [batch, M, N] -> stride [M * N, N, 1]
            *
            *   sum += A.data[(depth * M * K) + (row * K) + k]
            *        * B.data[(depth * K * N) + (k * N) + column];
            *
            *   C.data[(depth * M * N) + (row * N) + column] = sum;
            *
            * Com stride, a conta continua certa mesmo quando o tensor não é contíguo
            * (por exemplo, transposto ou com broadcast, em que stride[0] = 0).
         */
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += A.data[(depth * A.stride[0]) + (row * A.stride[1]) + (k      * A.stride[2])]     // A[depth][row][k]
                 * B.data[(depth * B.stride[0]) + (k   * B.stride[1]) + (column * B.stride[2])];    // B[depth][k][column]
        }
        C.data[(depth * C.stride[0]) + (row * C.stride[1]) + (column * C.stride[2])] = sum;       // C[depth][row][column]
    }

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
    Status matmul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C){
        size_t threads = 256;
        size_t blocks  = ((C->shape[0] * C->shape[1] * C->shape[2]) * threads - 1) / threads; // to-do

        // Precisamos criar uma função de tradução de TENSOR para TENSOR3
        size_t M = A->shape[1];
        size_t K = A->shape[2];
        size_t N = B->shape[2];

        matmul_kernel<<<blocks, threads>>>(*A, *B, *C, M, K, N);

        return cudaGetLastError() == cudaSuccess ? TENSOR_SUCCESS : TENSOR_EXEC_FAILED;
    }

}
