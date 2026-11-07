#include "status.hpp"


namespace tensor_status {

    static const char* const STATUS_NAMES[TENSOR_STATUS_COUNT] = {
        [TENSOR_SUCCESS]       = "SUCCESS",
        [TENSOR_BAD_PARAM]     = "INVALID PARAM",
        [TENSOR_NOT_SUPPORTED] = "NOT SUPPORTED",
        [TENSOR_EXEC_FAILED]   = "EXECUTION FAILED",
    };

    const char* status_name(Status s){
        return (size_t)s < TENSOR_STATUS_COUNT && STATUS_NAMES[s] != NULL ? STATUS_NAMES[s] : "UNRECOGNIZED STATUS"; 
    } 
}