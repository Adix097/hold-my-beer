#include "ctorch.h"
#include <string.h>

// returns the total number of elements by multiplying the sizes
// a zero dimensional tensor contains one element
uint64_t ct_tensor_numel(const CTTensor* t) {
    uint64_t n = 1;
    for (size_t i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

// computes row-major strides for a contiguous tensor
void compute_contiguous_strides(size_t ndim, const uint64_t* shape, uint64_t* strides) {
    uint64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
}

// returns a tensor with all elements as zero
CTTensor* ct_tensor_zeros(size_t ndim, const uint64_t* shape) {
    CT_CHECK(ndim <= CT_MAX_DIMS, "ndim out of range");

    CTTensor* t = calloc(1, sizeof* t);
    t->refcount = 1;
    CT_CHECK(t != NULL, "out of memory");

    t->ndim = ndim;
    uint64_t numel = 1;
    for (size_t i = 0; i < ndim; i++) {
        CT_CHECK(shape[i] != UINT64_MAX, "negative dimension size");
        t->shape[i] = shape[i];
        numel *= shape[i];
    }
    compute_contiguous_strides(ndim, shape, t->strides);
    t->offset = 0;
    t->storage = ct_storage_new((size_t)numel);
    return t;
}

// returns a tensor with all elements as one
CTTensor* ct_tensor_ones(size_t ndim, const uint64_t* shape) {
    CTTensor* t = ct_tensor_zeros(ndim, shape);
    uint64_t n = ct_tensor_numel(t);
    for (uint64_t i = 0; i < n; i++) t->storage->data[i] = 1.0f;
    return t;
}

// returns a tensor with specified shape and data
CTTensor* ct_tensor_from_data(const float* data, size_t ndim, const uint64_t* shape) {
    CTTensor* t = ct_tensor_zeros(ndim, shape);
    memcpy(t->storage->data, data, (size_t)ct_tensor_numel(t)*  sizeof(float));
    return t;
}

// retains the tensor by incrementing its reference count
CTTensor* ct_tensor_retain(CTTensor* t) {
    t->refcount++;
    return t;
}

// destroy the tensor
void ct_tensor_free(CTTensor* t) {
    if (t == NULL) return;
    CT_CHECK(t->refcount > 0, "freeing tensor with refcount <= 0");
    if (--t->refcount > 0) return;
    
    ct_storage_release(t->storage);
    ct_tensor_free(t->grad);
    ct_node_free(t->grad_fn);
    free(t);
}

// returns the index for each element in tensor as if it were a flat array
// pos = offset + sum(idx[i] * strides[i] for i in range(ndim))
static uint64_t flat_index(const CTTensor* t, const uint64_t* idx) {
    uint64_t pos = t->offset;
    for (size_t i = 0; i < t->ndim; i++) {
        CT_CHECK(idx[i] < t->shape[i], "index out of bounds");
        pos += idx[i] * t->strides[i];
    }
    return pos;
}

float ct_tensor_get(const CTTensor* t, const uint64_t* idx) {
    return t->storage->data[flat_index(t, idx)];
}

void ct_tensor_set(CTTensor* t, const uint64_t* idx, float value) {
    t->storage->data[flat_index(t, idx)] = value;
}

static void print_rec(const CTTensor* t, size_t dim, uint64_t pos, int indent) {
    if (dim == t->ndim) {
        printf("%.4f", t->storage->data[pos]);
        return;
    }
    printf("[");
    for (uint64_t i = 0; i < t->shape[dim]; i++) {
        if (dim == t->ndim - 1) {
            printf("%8.4f", t->storage->data[pos + i * t->strides[dim]]);
            if (i + 1 < t->shape[dim]) printf(", ");
        } else {
            if (i > 0) {
                printf(",\n");
                for (int k = 0; k < indent + 1; k++) printf(" ");
            }
            print_rec(t, dim + 1, pos + i * t->strides[dim], indent + 1);
        }
    }
    printf("]");
}

void ct_tensor_print(const CTTensor* t) {
    printf("tensor(");
    print_rec(t, 0, t->offset, 7);
    printf(", shape=[");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%llu%s", (unsigned long long)t->shape[i], i + 1 < t->ndim ? ", " : "");
    }
    printf("] , strides=[");

    for (size_t i = 0; i < t->ndim; i++) {
        printf("%llu%s", (unsigned long long)t->strides[i], i + 1 < t->ndim ? ", " : "");
    }
    printf("])\n");
}