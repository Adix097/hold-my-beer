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

int ct_grad_enabled = 1;

void ct_record(CTTensor* out, const char* name, const CTTensor* a, const CTTensor* b, CTBackwardFn backward) {
    if (!ct_grad_enabled) return;
    int needs = a->requires_grad || (b != NULL && b->requires_grad);
    if (!needs) return;
    out->requires_grad = 1;
    out->grad_fn = ct_node_new(name, a, b);
    out->grad_fn->backward = backward;
}

// shrink a gradient back down to the shape of the tensor it belongs to
// returns a tensor the caller owns
static CTTensor* sum_to_shape(const CTTensor* g, const CTTensor* target) {
    CTTensor* cur = ct_tensor_retain((CTTensor*)g);

    while (cur->ndim > target->ndim) {
        CTTensor* next = ct_tensor_sum(cur, 0, 0);
        ct_tensor_free(cur);
        cur = next;
    }

    for (size_t i = 0; i < target->ndim; i++) { // size-1 dims that were stretched
        if (target->shape[i] == 1 && cur->shape[i] != 1) {
            CTTensor* next = ct_tensor_sum(cur, i, 1);
            ct_tensor_free(cur);
            cur = next;
        }
    }

    return cur;
}

void ct_backward_add(CTNode* node, const CTTensor* g, CTTensor** grads_in) {
    for (size_t i = 0; i < node->n_inputs; i++) {
        CTTensor* in = node->inputs[i];
        grads_in[i] = in->requires_grad ? sum_to_shape(g, in) : NULL;
    }
}

void ct_backward_mul(CTNode* node, const CTTensor* g, CTTensor** grads_in) {
    CTTensor* a = node->inputs[0];
    CTTensor* b = node->inputs[1];
    grads_in[0] = grads_in[1] = NULL;

    if (a->requires_grad) {
        CTTensor* full = ct_tensor_mul(g, b); // g * b, in the broadcast shape
        grads_in[0] = sum_to_shape(full, a);
        ct_tensor_free(full);
    }

    if (b->requires_grad) {
        CTTensor* full = ct_tensor_mul(g, a);
        grads_in[1] = sum_to_shape(full, b);
        ct_tensor_free(full);
    }
}

// ---------- The backward walk ----------

typedef struct {
    CTTensor** items;
    size_t len;
    size_t cap;
} TensorList;

static void list_push(TensorList* l, CTTensor* t) {
    if (l->len == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->items = realloc(l->items, l->cap * sizeof *l->items);
        CT_CHECK(l->items != NULL, "out of memory (topo list)");
    }
    l->items[l->len++] = t;
}

// depth first search
static void topo_visit(CTTensor* t, TensorList* order) {
    if (t->grad_fn == NULL || t->grad_fn->visited) return;
    t->grad_fn->visited = 1;
    for (size_t i = 0; i < t->grad_fn->n_inputs; i++)
        topo_visit(t->grad_fn->inputs[i], order);
    list_push(order, t);
}

// takes ownership of g adds it into t->grad (never modifies in place)
static void accumulate(CTTensor* t, CTTensor* g) {
    if (t->grad == NULL) {
        t->grad = g;
    } else {
        CTTensor* sum = ct_tensor_add(t->grad, g);
        ct_tensor_free(t->grad);
        ct_tensor_free(g);
        t->grad = sum;
    }
}

void ct_tensor_backward(CTTensor* root) {
    CT_CHECK(root->requires_grad, "backward: tensor does not require grad");

    TensorList order = {0};
    topo_visit(root, &order);

    ct_grad_enabled = 0; // don't build graphs while computing gradients
    accumulate(root, ct_tensor_ones(root->ndim, root->shape));

    for (size_t i = order.len; i-- > 0;) { // reverse order because outputs first
        CTTensor* t = order.items[i];
        CTNode* n = t->grad_fn;
        CT_CHECK(n->backward != NULL, "backward not implemented for this op");
        CT_CHECK(t->grad != NULL, "backward: missing gradient");

        CTTensor* grads_in[CT_MAX_INPUTS] = {NULL};
        n->backward(n, t->grad, grads_in);
        for (size_t j = 0; j < n->n_inputs; j++) {
            if (grads_in[j] != NULL) accumulate(n->inputs[j], grads_in[j]);
        }
    }

    for (size_t i = 0; i < order.len; i++) order.items[i]->grad_fn->visited = 0;
    free(order.items);
    ct_grad_enabled = 1;
}