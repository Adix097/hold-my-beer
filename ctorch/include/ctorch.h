#ifndef CTORCH_H
#define CTORCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CT_MAX_DIMS 8
#define CT_MAX_INPUTS 2

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

typedef struct CTNode CTNode;

typedef struct CTTensor {
    CTStorage* storage;
    size_t ndim;
    size_t shape[CT_MAX_DIMS];
    size_t strides[CT_MAX_DIMS];
    size_t offset; // starting position in the storage

    int refcount;
    int requires_grad;
    struct CTTensor* grad; // NULL until backward fills it
    CTNode* grad_fn; // NULL for tensors created by user
} CTTensor;

struct CTNode {
    const char* name;
    size_t n_inputs;
    CTTensor* inputs[CT_MAX_INPUTS];
    int visited; // needed for backward walk
    size_t dim;
    int keepdim;
    void (*backward)(CTNode* node, const CTTensor* grad_out, CTTensor** grad_in);
};

void compute_contiguous_strides(size_t ndim, const uint64_t* shape, uint64_t* strides);

CTTensor* ct_tensor_zeros(size_t ndim, const uint64_t* shape);
CTTensor* ct_tensor_ones(size_t ndim, const uint64_t* shape);
CTTensor* ct_tensor_from_data(const float* data, size_t ndim, const uint64_t* shape);
CTTensor* ct_tensor_retain(CTTensor* t);
void ct_tensor_free(CTTensor* t);

CTTensor* ct_tensor_transpose(const CTTensor* t, size_t dim0, size_t dim1);
CTTensor* ct_tensor_permute(const CTTensor* t, const size_t* dims);
CTTensor* ct_tensor_slice(const CTTensor* t, size_t dim, size_t start, size_t end, size_t step);
CTTensor* ct_tensor_select(const CTTensor* t, size_t dim, size_t index);
int ct_tensor_is_contiguous(const CTTensor* t);
CTTensor* ct_tensor_contiguous(const CTTensor* t);
CTTensor* ct_tensor_reshape(const CTTensor* t, size_t ndim, const size_t* new_shape);

uint64_t ct_tensor_numel(const CTTensor* t);
float ct_tensor_get(const CTTensor* t, const uint64_t* idx);
void ct_tensor_set(CTTensor* t, const uint64_t* idx, float value);
void ct_tensor_print(const CTTensor* t);

CTTensor* ct_tensor_expand(const CTTensor* t, size_t ndim, const size_t* shape);

CTTensor* ct_tensor_add(const CTTensor* a, const CTTensor* b);
CTTensor* ct_tensor_sub(const CTTensor* a, const CTTensor* b);
CTTensor* ct_tensor_mul(const CTTensor* a, const CTTensor* b);
CTTensor* ct_tensor_div(const CTTensor* a, const CTTensor* b);

CTTensor* ct_tensor_neg(const CTTensor* t);
CTTensor* ct_tensor_exp(const CTTensor* t);
CTTensor* ct_tensor_log(const CTTensor* t);
CTTensor* ct_tensor_sqrt(const CTTensor* t);
CTTensor* ct_tensor_relu(const CTTensor* t);

CTTensor* ct_tensor_sum(const CTTensor* t, size_t dim, int keepdim);
CTTensor* ct_tensor_mean(const CTTensor* t, size_t dim, int keepdim);
CTTensor* ct_tensor_max(const CTTensor* t, size_t dim, int keepdim);
CTTensor* ct_tensor_sum_all(const CTTensor* t);
CTTensor* ct_tensor_matmul(const CTTensor* a, const CTTensor* b);

CTNode* ct_node_new(const char* name, const CTTensor* a, const CTTensor* b);
void ct_node_free(CTNode* n);

typedef void (*CTBackwardFn)(CTNode*, const CTTensor*, CTTensor**);

extern int ct_grad_enabled;
void ct_record(CTTensor* out, const char* name, const CTTensor* a, const CTTensor* b, CTBackwardFn backward);

void ct_backward_add(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_mul(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_tensor_backward(CTTensor* root);

void ct_backward_sum(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_mean(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_sub(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_div(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_neg(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);

void ct_backward_exp(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_log(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_sqrt(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);
void ct_backward_relu(CTNode* node, const CTTensor* grad_out, CTTensor** grads_in);

#endif