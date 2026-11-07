#include "device/layout.hpp"
#include "status.hpp"

using namespace tensor_status;

namespace DEVICE {
        static Status to_shape3(const size_t* shape, size_t rank, size_t shape3[3]){
            if(rank < 2) return TENSOR_NOT_SUPPORTED;
            size_t depth = 1;

            for(size_t i = 0; i < rank - 2; i++){
                depth *= shape[i];
            }

            shape3[0] = depth;
            shape3[1] = shape[rank - 2];
            shape3[2] = shape[rank - 1];

            return TENSOR_SUCCESS;
        }

        Status to_tensor3(const TENSOR* in, TENSOR3* out){
            // convert +3D-rank shape into 3D44

            // handle 2D rank as 3D

            // stride 3D: derivar do stride original (não recalcular)
            //   rank 2:  [0,           stride[0],    stride[1]]
            //   rank 3:  [stride[0],   stride[1],    stride[2]]
            //   rank >3: [stride[r-3], stride[r-2],  stride[r-1]]  (se o batch for contíguo)

        }
}
