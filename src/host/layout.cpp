#include "tensor/host/layout.hpp"
#include "tensor/status.hpp"

using namespace tensor_status;

namespace HOST {
        Status validate_matmul(const TENSOR* A, const TENSOR* B){
            /*
             * shape = [bloco, linha, coluna]
             *
             * A: shape = [2, 2, 3], stride = [ 6, 3, 1]
             * B: shape = [2, 3, 4], stride = [12, 4, 1]
             * C: shape = [2, 2, 4], stride = [ 8, 4, 1]
             */

            if(A == nullptr || B == nullptr || A->shape == nullptr || B->shape == nullptr)
                return TENSOR_MATMUL_NULL_PARAM;

            // shape[rank - 2] com rank < 2 daria a volta no size_t
            if(A->rank < 2 || B->rank < 2) return TENSOR_RANK_TOO_SMALL;

            // Se as colunas de A forem diferentes das linhas de B
            if(A->shape[A->rank - 1] != B->shape[B->rank - 2]){
                return TENSOR_MATMUL_K_MISMATCH;
            }

            return TENSOR_SUCCESS;
        }

        Status validate_operations(const TENSOR* A, const TENSOR* B){
            Status st = validate_matmul(A, B);
            if(st != TENSOR_SUCCESS) return st;

            return TENSOR_SUCCESS;
        }
}
