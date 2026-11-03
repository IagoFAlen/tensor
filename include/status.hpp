#pragma once

#include <cstddef>

namespace tensor_status {
    typedef enum Status {
        TENSOR_OK,
        TENSOR_BAD_PARAM,
        TENSOR_NOT_SUPPORTED,
        TENSOR_EXEC_FAILED,
        TENSOR_STATUS_COUNT
    } Status;
    
    const char* status_name(Status s);    
}
