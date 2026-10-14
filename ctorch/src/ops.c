#include "ctorch.h"
#include <math.h>

// Align shapes from the right; each pair must match or contain a 1
static size_t broadcast_shape(const CTTensor* a, const CTTensor* b, size_t* out) {
    size_t nd = a->ndim > b->ndim ? a->ndim : b->ndim;

    for (size_t i = 0; i < nd; i++) {
        size_t da = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
        size_t db = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
        CT_CHECK(da == db || da == 1 || db == 1, "shapes not broadcastable");
        out[nd - 1 - i] = (da == 1) ? db : da;
    }
    return nd;
}

#define CT_DEFINE_BINARY(name, expr)                                      \
    CTTensor* name(const CTTensor* a, const CTTensor* b) {                \
        size_t shape[CT_MAX_DIMS];                                        \
        size_t ndim = broadcast_shape(a, b, shape);                       \
        CTTensor* ea = ct_tensor_expand(a, ndim, shape);                  \
        CTTensor* eb = ct_tensor_expand(b, ndim, shape);                  \
        CTTensor* out = ct_tensor_zeros(ndim, shape);                     \
        size_t n = ct_tensor_numel(out);                                  \
        size_t idx[CT_MAX_DIMS] = {0};                                    \
        size_t pa = ea->offset, pb = eb->offset;                          \
        for (size_t i = 0; i < n; i++) {                                  \
            float x = ea->storage->data[pa];                              \
            float y = eb->storage->data[pb];                              \
            out->storage->data[i] = (expr);                               \
            for (size_t d = ndim; d-- > 0;) {                             \
                idx[d]++;                                                 \
                pa += ea->strides[d];                                     \
                pb += eb->strides[d];                                     \
                if (idx[d] < shape[d]) break;                             \
                pa -= ea->strides[d] * shape[d];                          \
                pb -= eb->strides[d] * shape[d];                          \
                idx[d] = 0;                                               \
            }                                                             \
        }                                                                 \
        ct_tensor_free(ea);                                               \
        ct_tensor_free(eb);                                               \
        return out;                                                       \
    }

CT_DEFINE_BINARY(ct_tensor_add, x + y)
CT_DEFINE_BINARY(ct_tensor_sub, x - y)
CT_DEFINE_BINARY(ct_tensor_mul, x * y)
CT_DEFINE_BINARY(ct_tensor_div, x / y)

#define CT_DEFINE_UNARY(name, expr)                                       \
    CTTensor* name(const CTTensor* t) {                                   \
        CTTensor* src = ct_tensor_contiguous(t);                          \
        CTTensor* out = ct_tensor_zeros(t->ndim, t->shape);               \
        size_t n = ct_tensor_numel(out);                                  \
        const float* in = src->storage->data + src->offset;               \
        for (size_t i = 0; i < n; i++) {                                  \
            float x = in[i];                                              \
            out->storage->data[i] = (expr);                               \
        }                                                                 \
        ct_tensor_free(src);                                              \
        return out;                                                       \
    }

CT_DEFINE_UNARY(ct_tensor_neg, -x)
CT_DEFINE_UNARY(ct_tensor_exp, expf(x))
CT_DEFINE_UNARY(ct_tensor_log, logf(x))
CT_DEFINE_UNARY(ct_tensor_sqrt, sqrtf(x))
CT_DEFINE_UNARY(ct_tensor_relu, x > 0.0f ? x : 0.0f)