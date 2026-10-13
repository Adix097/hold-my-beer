#ifndef CTORCH_H
#define CTORCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CT_MAX_DIMS 8

// minimal error handling: prints location and abort
#define CT_CHECK(cond, msg)                                                            \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "CTorch error: %s (%s:%d)\n", (msg), __FILE__, __LINE__);                           \
            abort();                                                                   \
        }                                                                              \
    } while (0)

typedef struct CTStorage {
    float* data;
    size_t numel;
    int refcount;
} CTStorage;

CTStorage* ct_storage_new(size_t numel); // ref count starts at 1
CTStorage* ct_storage_retain(CTStorage* s);
void ct_storage_release(CTStorage* s);

typedef struct CTTensor {
    CTStorage* storage;
    int ndim;
    int64_t shape[CT_MAX_DIMS];
    int64_t strides[CT_MAX_DIMS];
    int64_t offset; // start position in storage
} CTTensor;

CTTensor* ct_tensor_zeros(int ndim, const int64_t* shape);
CTTensor* ct_tensor_ones(int ndim, const int64_t* shape);
CTTensor* ct_tensor_from_data(const float* data, int ndim, const int64_t* shape);
void ct_tensor_free(CTTensor* t);

int64_t ct_tensor_numel(const CTTensor* t);
float ct_tensor_get(const CTTensor* t, const int64_t* idx);
void ct_tensor_set(CTTensor* t, const int64_t* idx, float value);
void ct_tensor_print(const CTTensor* t);

#endif