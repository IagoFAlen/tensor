#include "status.hpp"


namespace tensor_status {

    /*
     * Um case por status. Sem default: se um status novo for criado e não ganhar
     * nome aqui, o compilador avisa (-Wswitch).
     */
    const char* status_name(Status s){
        switch(s){
            case TENSOR_SUCCESS:                 return "SUCCESS";

            case TENSOR_BAD_PARAM:               return "INVALID PARAM";
            case TENSOR_NOT_SUPPORTED:           return "NOT SUPPORTED";
            case TENSOR_EXEC_FAILED:             return "EXECUTION FAILED";

            case TENSOR_ALLOC_FAILED:            return "ALLOC: DEVICE ALLOCATION FAILED";
            case TENSOR_FEED_NOT_ALLOCATED:      return "FEED: DEVICE TENSOR NOT ALLOCATED";
            case TENSOR_FEED_NULL_DATA:          return "FEED: NULL HOST DATA";
            case TENSOR_COPY_FAILED:             return "FEED: HOST TO DEVICE COPY FAILED";
            case TENSOR_FETCH_NOT_ALLOCATED:     return "FETCH: DEVICE TENSOR NOT ALLOCATED";
            case TENSOR_FETCH_NULL_DATA:         return "FETCH: NULL HOST DATA";
            case TENSOR_FETCH_COPY_FAILED:       return "FETCH: DEVICE TO HOST COPY FAILED";

            case TENSOR_HOST_ALLOC_FAILED:       return "ALLOC: HOST ALLOCATION FAILED";

            case TENSOR_LAYOUT_NULL_PARAM:       return "LAYOUT: NULL PARAM";
            case TENSOR_RANK_TOO_SMALL:          return "LAYOUT: RANK SMALLER THAN 2";
            case TENSOR_BATCH_NOT_CONTIGUOUS:    return "LAYOUT: BATCH NOT CONTIGUOUS";
            case TENSOR_TRANSPOSE_NULL_PARAM:    return "TRANSPOSE: NULL PARAM";

            case TENSOR_MATMUL_NULL_PARAM:       return "MATMUL: NULL PARAM";
            case TENSOR_MATMUL_K_MISMATCH:       return "MATMUL: COLUMNS OF A != ROWS OF B";
            case TENSOR_MATMUL_BATCH_MISMATCH:   return "MATMUL: INCOMPATIBLE BATCH";
            case TENSOR_MATMUL_C_SHAPE:          return "MATMUL: C SHAPE != [BATCH, M, N]";

            case TENSOR_HADAMARD_NULL_PARAM:     return "HADAMARD: NULL PARAM";
            case TENSOR_HADAMARD_SHAPE_MISMATCH: return "HADAMARD: ROWS OR COLUMNS OF A/B != C";
            case TENSOR_HADAMARD_BATCH_MISMATCH: return "HADAMARD: INCOMPATIBLE BATCH";

            case TENSOR_LAUNCH_FAILED:           return "KERNEL LAUNCH FAILED";

            case TENSOR_STATUS_COUNT:            break;
        }
        return "UNRECOGNIZED STATUS";
    }
}
