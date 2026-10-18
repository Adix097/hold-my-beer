#include "ctorch.h"

// b may be NULL for ops with one input
CTNode* ct_node_new(const char* name, const CTTensor* a, const CTTensor* b) {
    CTNode* n = calloc(1, sizeof *n);
    CT_CHECK(n != NULL, "out of memory (node)");
    n->name = name;
    // retain should take a non-const pointer because it changes the refcount
    // the tensor's data isn't touched
    n->inputs[n->n_inputs++] = ct_tensor_retain((CTTensor*)a);
    if (b != NULL) n->inputs[n->n_inputs++] = ct_tensor_retain((CTTensor*)b);
    return n;
}

void ct_node_free(CTNode* n) {
    if (n == NULL) return;
    for (size_t i = 0; i < n->n_inputs; i++) ct_tensor_free(n->inputs[i]);
    free(n);
}

void ct_record(CTTensor* out, const char* name, const CTTensor* a, const CTTensor* b) {
    int needs = a->requires_grad || (b != NULL && b->requires_grad);
    if (!needs) return;
    out->requires_grad = 1;
    out->grad_fn = ct_node_new(name, a, b);
}