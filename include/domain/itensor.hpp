
#ifndef ITENSOR_H
#define ITENSOR_H
#pragma once

#include <iostream>

using namespace std;

typedef struct ITensor {
    void  (*add_vertexes)(void* self);
    void  (*add_block)(void* self, float* weights_block, size_t block_size);
    float (*get_weight_by_coord)(void* self, size_t* coord);
    void  (*destroy)(void* self);
};

#endif