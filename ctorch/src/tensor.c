#include "ctorch.h"
#include <string.h>

/*
* functions declared as static are meant of internal use only
* no other .c file can call those functions
*/

// returns the total number of elements by multiplying the sizes
// a zero dimensional tensor contains one element
int64_t ct_tensor_numel(const CTTensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

// computes row-major strides for a contiguous tensor
static void compute_contiguous_strides(int ndim, const int64_t* shape, int64_t* strides) {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
}

// returns a tensor with all elements as zero
CTTensor* ct_tensor_zeros(int ndim, const int64_t* shape) {
    CT_CHECK(ndim >= 0 && ndim <= CT_MAX_DIMS, "ndim out of range");

    CTTensor* t = calloc(1, sizeof* t);
    CT_CHECK(t != NULL, "out of memory");

    t->ndim = ndim;
    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        CT_CHECK(shape[i] >= 0, "negative dimension size");
        t->shape[i] = shape[i];
        numel *= shape[i];
    }
    compute_contiguous_strides(ndim, shape, t->strides);
    t->offset = 0;
    t->storage = ct_storage_new((size_t)numel);
    return t;
}

// returns a tensor with all elements as one
CTTensor* ct_tensor_ones(int ndim, const int64_t* shape) {
    CTTensor* t = ct_tensor_zeros(ndim, shape);
    int64_t n = ct_tensor_numel(t);
    for (int64_t i = 0; i < n; i++) t->storage->data[i] = 1.0f;
    return t;
}

// returns a tensor with specified shape and data
CTTensor* ct_tensor_from_data(const float* data, int ndim, const int64_t* shape) {
    CTTensor* t = ct_tensor_zeros(ndim, shape);
    memcpy(t->storage->data, data, (size_t)ct_tensor_numel(t)*  sizeof(float));
    return t;
}

// destroy the tensor
void ct_tensor_free(CTTensor* t) {
    if (t == NULL) return;
    ct_storage_release(t->storage);
    free(t);
}

// returns the index for each element in tensor as if it were a flat array
// pos = offset + sum(idx[i] * strides[i] for i in range(ndim))
static int64_t flat_index(const CTTensor* t, const int64_t* idx) {
    int64_t pos = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        CT_CHECK(idx[i] >= 0 && idx[i] < t->shape[i], "index out of bounds");
        pos += idx[i] * t->strides[i];
    }
    return pos;
}

float ct_tensor_get(const CTTensor* t, const int64_t* idx) {
    return t->storage->data[flat_index(t, idx)];
}

void ct_tensor_set(CTTensor* t, const int64_t* idx, float value) {
    t->storage->data[flat_index(t, idx)] = value;
}

static void print_rec(const CTTensor* t, int dim, int64_t pos, int indent) {
    if (dim == t->ndim) { 
        printf("%.4f", t->storage->data[pos]);
        return;
    }
    printf("[");
    for (int64_t i = 0; i < t->shape[dim]; i++) {
        if (dim == t->ndim - 1) {
            printf("%8.4f", t->storage->data[pos + i*  t->strides[dim]]);
            if (i + 1 < t->shape[dim]) printf(", ");
        } else {
            if (i > 0) {
                printf(",\n");
                for (int k = 0; k < indent + 1; k++) printf(" ");
            }
            print_rec(t, dim + 1, pos + i*  t->strides[dim], indent + 1);
        }
    }
    printf("]");
}

void ct_tensor_print(const CTTensor* t) {
    printf("tensor(");
    print_rec(t, 0, t->offset, 7);
    printf(", shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", (long long)t->shape[i], i + 1 < t->ndim ? ", " : "");
    }
    printf("], strides=[");
    
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", (long long)t->strides[i], i + 1 < t->ndim ? ", " : "");
    }    
    printf("])\n");
}