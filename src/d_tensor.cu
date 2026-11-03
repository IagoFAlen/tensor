#include "d_tensor.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
    __global__ void matmul_kernel(){
        
    }

    Status matmul(const TENSOR* A, const TENSOR* B, TENSOR* C){
        size_t threads = 256;
        size_t blocks  = 0; // to-do
        
        matmul_kernel<<<blocks, threads>>>();

        return cudaGetLastError() == cudaSuccess ? TENSOR_OK : TENSOR_EXEC_FAILED;
    }


}
