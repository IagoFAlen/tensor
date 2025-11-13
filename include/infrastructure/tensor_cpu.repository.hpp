
#include "tensor.hpp"

#include <cstring>
#include <cstdio>

namespace cpu_tensor {
    static void add_vertexes(void* self);
    static void add_weights(void* self, size_t* coord, float weight);
    static float get_weight(void* self, size_t* coord);
    static void destroy(void* self);
    ITensor* create_tensor(size_t rank, size_t* dimensions_size, float default_weight);
}