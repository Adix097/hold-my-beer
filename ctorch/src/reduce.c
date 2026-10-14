#include "ctorch.h"
#include <math.h>

// split the tensor around dim into outer, reduce, inner
static CTTensor* reduce_alloc(const CTTensor* t, size_t dim, int keepdim, size_t* outer, size_t* red, size_t* inner) {
    CT_CHECK(dim < t->ndim, "reduce: dim out of range");
    *outer = 1;
    *inner = 1;
    for (size_t i = 0; i < dim; i++) *outer *= t->shape[i];
    for (size_t i = dim + 1; i < t->ndim; i++) *inner *= t->shape[i];
    *red = t->shape[dim];

    size_t shape[CT_MAX_DIMS];
    size_t nd = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) shape[nd++] = 1;
        } else {
            shape[nd++] = t->shape[i];
        }
    }
    return ct_tensor_zeros(nd, shape);
}

#define CT_DEFINE_REDUCE(name, init, update)                              \
    CTTensor* name(const CTTensor* t, size_t dim, int keepdim) {          \
        size_t outer, red, inner;                                         \
        CTTensor* out = reduce_alloc(t, dim, keepdim, &outer, &red, &inner); \
        CTTensor* src = ct_tensor_contiguous(t);                          \
        const float* in = src->storage->data + src->offset;               \
        float* o = out->storage->data;                                    \
        for (size_t a = 0; a < outer; a++) {                              \
            for (size_t b = 0; b < inner; b++) {                          \
                float acc = (init);                                       \
                for (size_t r = 0; r < red; r++) {                        \
                    float x = in[(a * red + r) * inner + b];              \
                    acc = (update);                                       \
                }                                                         \
                o[a * inner + b] = acc;                                   \
            }                                                             \
        }                                                                 \
        ct_tensor_free(src);                                              \
        return out;                                                       \
    }

CT_DEFINE_REDUCE(ct_tensor_sum, 0.0f, acc + x)
CT_DEFINE_REDUCE(ct_tensor_max, -INFINITY, x > acc ? x : acc)

CTTensor* ct_tensor_mean(const CTTensor* t, size_t dim, int keepdim) {
    CTTensor* out = ct_tensor_sum(t, dim, keepdim);
    float scale = 1.0f / (float)t->shape[dim];
    size_t n = ct_tensor_numel(out);
    for (size_t i = 0; i < n; i++) out->storage->data[i] *= scale;
    return out;
}

CTTensor* ct_tensor_sum_all(const CTTensor* t) {
    CTTensor* src = ct_tensor_contiguous(t);
    CTTensor* out = ct_tensor_zeros(0, NULL);
    const float* in = src->storage->data + src->offset;
    size_t n = ct_tensor_numel(src);
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) acc += in[i];
    out->storage->data[0] = acc;
    ct_tensor_free(src);
    return out;
}

CTTensor* ct_tensor_matmul(const CTTensor* a, const CTTensor* b) {
    CT_CHECK(a->ndim == 2 && b->ndim == 2, "matmul: only 2-d tensors for now");
    CT_CHECK(a->shape[1] == b->shape[0], "matmul: inner dimensions differ");
    size_t M = a->shape[0], K = a->shape[1], N = b->shape[1];

    CTTensor* ac = ct_tensor_contiguous(a);
    CTTensor* bc = ct_tensor_contiguous(b);
    size_t shape[] = {M, N};
    CTTensor* out = ct_tensor_zeros(2, shape);

    const float* A = ac->storage->data + ac->offset;
    const float* B = bc->storage->data + bc->offset;
    float* C = out->storage->data;

    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float aik = A[i * K + k];
            for (size_t j = 0; j < N; j++) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
    ct_tensor_free(ac);
    ct_tensor_free(bc);
    return out;
}