
#ifndef ITENSOR_H
#define ITENSOR_H
#pragma once

#include <iostream>

using namespace std;

typedef struct ITensor {
    void  (*add_vertexes)(void* self);
    void  (*add_weights)(void* self, float* weight);
    void  (*add_weights_by_coord)(void* self, size_t* coord, float weight);
    float (*get_weight_by_coord)(void* self, size_t* coord);
    void  (*destroy)(void* self);
};

#endif