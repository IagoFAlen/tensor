#include "tensor/status.hpp"


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
            case TENSOR_FEED_NULL_DATA:          return "FEED: NULL OR UNALLOCATED HOST TENSOR";
            case TENSOR_FEED_SHAPE_MISMATCH:     return "FEED: HOST SHAPE != DEVICE SHAPE";
            case TENSOR_FEED_STRIDE_MISMATCH:    return "FEED: HOST STRIDE != DEVICE STRIDE";
            case TENSOR_HOST_TO_DEVICE_NOT_ALLOCATED: return "HOST TO DEVICE: DEVICE TENSOR NOT ALLOCATED";
            case TENSOR_HOST_TO_DEVICE_NULL_DATA:     return "HOST TO DEVICE: NULL HOST DATA";
            case TENSOR_HOST_TO_DEVICE_COPY_FAILED:   return "HOST TO DEVICE: COPY FAILED";
            case TENSOR_DEVICE_TO_HOST_NOT_ALLOCATED: return "DEVICE TO HOST: DEVICE TENSOR NOT ALLOCATED";
            case TENSOR_DEVICE_TO_HOST_NULL_DATA:     return "DEVICE TO HOST: NULL HOST DATA";
            case TENSOR_DEVICE_TO_HOST_COPY_FAILED:   return "DEVICE TO HOST: COPY FAILED";

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
