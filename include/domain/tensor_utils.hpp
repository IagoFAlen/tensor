#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H
#pragma once

#include <cstddef>

namespace tensor_utils {
    size_t calc_weights_size(size_t* dimensions, size_t rank);
    bool valid_index(size_t idx, size_t weights_size);
}

#endif
