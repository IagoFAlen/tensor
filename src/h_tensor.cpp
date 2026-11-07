#include "h_tensor.hpp"
#include "status.hpp"

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

            // Se as colunas de A forem diferentes das linhas de B
            if(A->shape[A->rank - 1] != B->shape[B->rank - 2]){
                return TENSOR_BAD_PARAM;
            }

            return TENSOR_SUCCESS;
        }

        Status validate_operations(const TENSOR* A, const TENSOR* B){
            Status st = validate_matmul(A, B);
            if(st != TENSOR_SUCCESS) return st;

            return TENSOR_SUCCESS;
        }
}
