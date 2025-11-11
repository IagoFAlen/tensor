#include "domain/tensor_utils.hpp"

namespace tensor_utils {
    size_t calc_weights_size(size_t* dimensions, size_t rank) {
        size_t total = 1;
        for (size_t i = 0; i < rank; ++i)
            total *= dimensions[i];
        return total;
    }

    bool valid_index(size_t idx, size_t weights_size) {
        return idx < weights_size;
    }
}
