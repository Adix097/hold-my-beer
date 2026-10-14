#include "ctorch.h"

// returns new header with identical metadata
static CTTensor* view_of(const CTTensor* src) {
    CTTensor* v = malloc(sizeof *v);
    CT_CHECK(v != NULL, "out of memory (view struct)");
    *v = *src;
    ct_storage_retain(v->storage);
    return v;
}

CTTensor* ct_tensor_transpose(const CTTensor* t, size_t dim0, size_t dim1) {
    CT_CHECK(dim0 < t->ndim && dim1 < t->ndim, "transpose: dim out of range");
    CTTensor* v = view_of(t);
    size_t tmp = v->shape[dim0];
    v->shape[dim0] = v->shape[dim1];
    v->shape[dim1] = tmp;
    tmp = v->strides[dim0];
    v->strides[dim0] = v->strides[dim1];
    v->strides[dim1] = tmp;
    return v;
}

CTTensor* ct_tensor_permute(const CTTensor* t, const size_t* dims) {
    int seen[CT_MAX_DIMS] = {0};
    CTTensor* v = view_of(t);
    for (size_t i = 0; i < t->ndim; i++) {
        size_t d = dims[i];
        CT_CHECK(d < t->ndim, "permute: dim out of range");
        CT_CHECK(!seen[d], "permute: repeated dim");
        seen[d] = 1;
        v->shape[i] = t->shape[d];
        v->strides[i] = t->strides[d];
    }
    return v;
}

CTTensor* ct_tensor_slice(const CTTensor* t, size_t dim, size_t start, size_t end, size_t step) {
    CT_CHECK(dim < t->ndim, "slice: dim out of range");
    CT_CHECK(step > 0, "slice: step must be positive");
    CT_CHECK(start <= end && end <= t->shape[dim], "slice: bad range");
    CTTensor* v = view_of(t);
    v->offset += start * t->strides[dim];
    v->shape[dim] = (end - start + step - 1) / step;
    v->strides[dim] = t->strides[dim] * step;
    return v;
}

CTTensor* ct_tensor_select(const CTTensor* t, size_t dim, size_t index) {
    CT_CHECK(dim < t->ndim, "select: dim out of range");
    CT_CHECK(index < t->shape[dim], "select: index out of range");
    CTTensor* v = view_of(t);
    v->offset += index * t->strides[dim];
    // removes dimension d
    for (size_t i = dim; i + 1 < t->ndim; i++) {
        v->shape[i] = t->shape[i + 1];
        v->strides[i] = t->strides[i + 1];
    }
    v->ndim = t->ndim - 1;
    return v;
}

int ct_tensor_is_contiguous(const CTTensor* t) {
    if (ct_tensor_numel(t) == 0) return 1;
    size_t expected = 1;
    for (size_t i = t->ndim; i-- > 0;) {
        if (t->shape[i] == 1) continue;
        if (t->strides[i] != expected) return 0;
        expected *= t->shape[i];
    }
    return 1;
}

CTTensor* ct_tensor_contiguous(const CTTensor* t) {
    if (ct_tensor_is_contiguous(t)) return view_of(t);

    CTTensor* out = ct_tensor_zeros(t->ndim, t->shape);
    size_t n = ct_tensor_numel(t);
    size_t idx[CT_MAX_DIMS] = {0};

    for (size_t i = 0; i < n; i++) {
        size_t pos = t->offset;
        for (size_t d = 0; d < t->ndim; d++) pos += idx[d] * t->strides[d];
        out->storage->data[i] = t->storage->data[pos];

        for (size_t d = t->ndim; d-- > 0;) {
            if (++idx[d] < t->shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

CTTensor* ct_tensor_reshape(const CTTensor* t, size_t ndim, const size_t* new_shape) {
    CT_CHECK(ndim <= CT_MAX_DIMS, "reshape: ndim out of range");

    const size_t inferred_dim = (size_t)-1;
    size_t inferred_index = SIZE_MAX;
    size_t known_count = 1;
    size_t shape_copy[CT_MAX_DIMS] = {0};

    for (size_t i = 0; i < ndim; i++) {
        shape_copy[i] = new_shape[i];
        if (shape_copy[i] == inferred_dim) {
            CT_CHECK(inferred_index == SIZE_MAX, "reshape: multiple inferred dimensions");
            inferred_index = i;
        } else {
            known_count *= shape_copy[i];
        }
    }

    const size_t total_count = ct_tensor_numel(t);
    if (inferred_index != SIZE_MAX) {
        CT_CHECK(known_count != 0, "reshape: element count mismatch");
        CT_CHECK(total_count % known_count == 0, "reshape: element count mismatch");
        shape_copy[inferred_index] = total_count / known_count;
    } else {
        CT_CHECK(known_count == total_count, "reshape: element count mismatch");
    }

    // contiguous() gives the header we own
    CTTensor* out = ct_tensor_contiguous(t);
    out->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) out->shape[i] = shape_copy[i];
    compute_contiguous_strides((int)ndim, out->shape, out->strides);
    return out;
}

CTTensor* ct_tensor_expand(const CTTensor* t, size_t ndim, const size_t* shape) {
    CT_CHECK(ndim >= t->ndim && ndim <= CT_MAX_DIMS, "expand: bad ndim");
    CTTensor* v = view_of(t);
    size_t lead = ndim - t->ndim;
    v->ndim = ndim;
    
    for (size_t i = 0; i < ndim; i++) {
        v->shape[i] = shape[i];
        if (i < lead) {
            v->strides[i] = 0;
        } else if (t->shape[i - lead] == shape[i]) {
            v->strides[i] = t->strides[i - lead];
        } else {
            CT_CHECK(t->shape[i - lead] == 1, "expand: shapes not broadcastable");
            v->strides[i] = 0;
        }
    }
    return v;
}