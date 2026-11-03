#include "h_tensor.hpp"
#include "status.hpp"

using namespace tensor;

namespace HOST {
        Status validate_matmul(TENSOR A, TENSOR B){
            /*
             * shape = [bloco, linha, coluna]
             *
             * A: shape = [2, 2, 3], stride = [ 6, 3, 1]
             * B: shape = [2, 3, 4], stride = [12, 4, 1]
             * C: shape = [2, 2, 4], stride = [ 8, 4, 1]
             */

            // Se as colunas de A forem diferentes das linhas de B
            if(A.shape[A.rank - 1] != B.shape[B.rank - 2]){
                return TENSOR_BAD_PARAM;
            }

            return TENSOR_OK;
        }

        Status validate_batch(TENSOR A, TENSOR B){
            for (size_t i = 0; i < A.rank - 2; i++) {
                if (A.shape[i] != B.shape[i]) {
                    return TENSOR_BAD_PARAM;
                }
            }

            return TENSOR_OK;
        }

        Status validate_operations(TENSOR A, TENSOR B){
            Status st = validate_matmul(A, B);
            if(st != TENSOR_OK) return st;

            st = validate_batch(A, B);
            if(st != TENSOR_OK) return st;

            return TENSOR_OK;
        }
}
