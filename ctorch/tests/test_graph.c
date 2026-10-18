#include "ctorch.h"

static void show(const CTTensor* t, int depth) {
    for (int i = 0; i < depth * 2; i++) printf(" ");
    float v = t->storage->data[t->offset];
    if (t->grad_fn == NULL) {
        printf("leaf (value %.1f, requires_grad %d)\n", v, t->requires_grad);
        return;
    }
    printf("%s (value %.1f)\n", t->grad_fn->name, v);
    for (size_t i = 0; i < t->grad_fn->n_inputs; i++) {
        show(t->grad_fn->inputs[i], depth + 1);
    }
}

int main(void) {
    size_t shape[] = {1};
    float xv = 2.0f, wv = 3.0f;
    CTTensor* x = ct_tensor_from_data(&xv, 1, shape);
    CTTensor* w = ct_tensor_from_data(&wv, 1, shape);
    x->requires_grad = 1;
    w->requires_grad = 1;

    CTTensor* y = ct_tensor_mul(x, w);
    CTTensor* L = ct_tensor_add(y, x);

    printf("graph of L = (x*w) + x:\n");
    show(L, 0);

    printf("refcounts: x=%d (expect 3), w=%d (expect 2), y=%d (expect 2), L=%d (expect 1)\n", x->refcount, w->refcount, y->refcount, L->refcount);

    // tensors with no requires_grad get no graph
    float pv = 5.0f;
    CTTensor* p = ct_tensor_from_data(&pv, 1, shape);
    CTTensor* q = ct_tensor_add(p, p);
    printf("plain add: requires_grad=%d, grad_fn is %s\n", q->requires_grad, q->grad_fn == NULL ? "NULL" : "set");

    ct_tensor_free(y);
    ct_tensor_free(x);
    ct_tensor_free(w);
    printf("after freeing y, x, w ourselves, L still shows:\n");
    show(L, 0);

    ct_tensor_free(L); // cascades through the whole graph
    ct_tensor_free(p);
    ct_tensor_free(q);
    printf("done\n");
    return 0;
}