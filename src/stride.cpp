#include "tensor/stride.hpp"

namespace tensor_stride {
    void contiguous_stride(const size_t* shape, size_t rank, size_t* stride){
        if (rank == 0) return;

        stride[rank - 1] = 1;
        for (size_t i = rank - 1; i-- > 0; ) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }
}
