#include "device/ops.hpp"
#include "status.hpp"
#include "config/device.hpp"

using namespace tensor_status;

namespace DEVICE {
    __global__ void hadamard_mul_kernel(TENSOR3 A, TENSOR3 B, TENSOR3 C){
        size_t row      = blockIdx.y * blockDim.y + threadIdx.y;
        size_t column   = blockIdx.x * blockDim.x + threadIdx.x;
        size_t depth    = blockIdx.z * blockDim.z + threadIdx.z;

        if (depth >= C.shape[0] || row >= C.shape[1] || column >= C.shape[2]) return;

        float mul = 0.00f;
        mul = A.data[(depth * A.stride[0]) + (row * A.stride[1]) + (column * A.stride[2])] // A[depth][row][column]
            * B.data[(depth * B.stride[0]) + (row * B.stride[1]) + (column * B.stride[2])];// B[depth][row][column]
    
        C.data[(depth * C.stride[0]) + (row * C.stride[1]) + (column * C.stride[2])] = mul;       // C[depth][row][column]
    
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
     * O batch de X serve para C se for igual ao de C, ou se for broadcast:
     * shape[0] == 1 e stride[0] == 0 (o mesmo bloco para todo depth, ex.: um bias [M, N]).
     */
    static bool batch_fits(const TENSOR3* X, const TENSOR3* C){
        return X->shape[0] == C->shape[0] || (X->shape[0] == 1 && X->stride[0] == 0);
    }

    // Um bloco cobre um pedaço BLOCK_X × BLOCK_Y de uma matriz de C (tamanho em config/device.hpp).
    static const dim3 THREADS(BLOCK_X, BLOCK_Y, 1);

    // O grid arredonda para cima em x e y, e tem um bloco por matriz do batch em z.
    static dim3 grid_for(const TENSOR3* C){
        return dim3((C->shape[2] + THREADS.x - 1) / THREADS.x,     // colunas (N)
                    (C->shape[1] + THREADS.y - 1) / THREADS.y,     // linhas  (M)
                    C->shape[0]);                                  // batch
    }

    // Os três tensores existem e têm memória (um data nulo faria o kernel ler/escrever em nullptr).
    static bool all_allocated(const TENSOR3* A, const TENSOR3* B, const TENSOR3* C){
        return A != nullptr && B != nullptr && C != nullptr &&
               A->data != nullptr && B->data != nullptr && C->data != nullptr;
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
    Status mat_mul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C){
        if (!all_allocated(A, B, C)) return TENSOR_MATMUL_NULL_PARAM;

        size_t M = A->shape[1];
        size_t K = A->shape[2];
        size_t N = B->shape[2];

        if (B->shape[1] != K) return TENSOR_MATMUL_K_MISMATCH;
        if (C->shape[1] != M || C->shape[2] != N) return TENSOR_MATMUL_C_SHAPE;
        if (!batch_fits(A, C) || !batch_fits(B, C)) return TENSOR_MATMUL_BATCH_MISMATCH;

        matmul_kernel<<<grid_for(C), THREADS>>>(*A, *B, *C, M, K, N);

        return cudaGetLastError() == cudaSuccess ? TENSOR_SUCCESS : TENSOR_LAUNCH_FAILED;
    }

    Status hadamard_mul(const TENSOR3* A, const TENSOR3* B, TENSOR3* C){
        if (!all_allocated(A, B, C)) return TENSOR_HADAMARD_NULL_PARAM;

        if (A->shape[1] != C->shape[1] || A->shape[2] != C->shape[2] ||
            B->shape[1] != C->shape[1] || B->shape[2] != C->shape[2]) return TENSOR_HADAMARD_SHAPE_MISMATCH;

        if (!batch_fits(A, C) || !batch_fits(B, C)) return TENSOR_HADAMARD_BATCH_MISMATCH;

        hadamard_mul_kernel<<<grid_for(C), THREADS>>>(*A, *B, *C);

        return cudaGetLastError() == cudaSuccess ? TENSOR_SUCCESS : TENSOR_LAUNCH_FAILED;
    }

    /*
     * O lançamento é assíncrono: mat_mul e hadamard_mul voltam antes do kernel terminar,
     * e cudaGetLastError() só pega erro de lançamento. Um erro dentro do kernel (ex.: acesso
     * inválido) só aparece aqui. Depois dele, o contexto CUDA fica inutilizável.
     */
    Status synchronize(){
        return cudaDeviceSynchronize() == cudaSuccess ? TENSOR_SUCCESS : TENSOR_EXEC_FAILED;
    }

}
