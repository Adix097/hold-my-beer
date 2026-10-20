#include "ctorch.h"

static CTTensor* scalar(float v) {
    size_t one[] = {1};
    CTTensor* t = ct_tensor_from_data(&v, 1, one);
    t->requires_grad = 1;
    return t;
}

int main(void) {
    {
        size_t one[] = {1};
        float xv = 2.0f, wv = 3.0f;
        CTTensor* x = ct_tensor_from_data(&xv, 1, one);
        CTTensor* w = ct_tensor_from_data(&wv, 1, one);
        x->requires_grad = 1;
        w->requires_grad = 1;

        CTTensor* y = ct_tensor_mul(x, w);
        CTTensor* L = ct_tensor_add(y, x);
        ct_tensor_backward(L);
        printf("x.grad = %.1f (expect 4)\n", x->grad->storage->data[0]);
        printf("w.grad = %.1f (expect 2)\n", w->grad->storage->data[0]);
        printf("y.grad = %.1f (expect 1)\n", y->grad->storage->data[0]);

        float ra[] = {1, 2, 3, 4, 5, 6};
        size_t sa[] = {2, 3};
        float rb[] = {10, 20, 30};
        size_t sb[] = {3};
        CTTensor* a = ct_tensor_from_data(ra, 2, sa);
        CTTensor* b = ct_tensor_from_data(rb, 1, sb);
        a->requires_grad = 1;
        b->requires_grad = 1;
        CTTensor* out = ct_tensor_mul(a, b);
        ct_tensor_backward(out);
        printf("a.grad = "); ct_tensor_print(a->grad);
        printf("b.grad = "); ct_tensor_print(b->grad);

        ct_tensor_free(x);
        ct_tensor_free(w);
        ct_tensor_free(y);
        ct_tensor_free(L);
        ct_tensor_free(a);
        ct_tensor_free(b);
        ct_tensor_free(out);
    }

    {
        CTTensor* x = scalar(5.0f);
        CTTensor* w = scalar(2.0f);
        CTTensor* L = ct_tensor_sub(x, w);
        ct_tensor_backward(L);
        printf("sub: x.grad = %.4f (expect 1), w.grad = %.4f (expect -1)\n", x->grad->storage->data[0], w->grad->storage->data[0]);
        ct_tensor_free(x);
        ct_tensor_free(w);
        ct_tensor_free(L);

        x = scalar(6.0f);
        w = scalar(3.0f);
        L = ct_tensor_div(x, w);
        ct_tensor_backward(L);
        printf("div: x.grad = %.4f (expect 0.3333), w.grad = %.4f (expect -0.6667)\n", x->grad->storage->data[0], w->grad->storage->data[0]);
        ct_tensor_free(x);
        ct_tensor_free(w);
        ct_tensor_free(L);

        x = scalar(2.0f);
        w = scalar(3.0f);
        CTTensor* y = ct_tensor_mul(x, w);
        L = ct_tensor_neg(y);
        ct_tensor_backward(L);
        printf("neg: x.grad = %.1f (expect -3), w.grad = %.1f (expect -2)\n", x->grad->storage->data[0], w->grad->storage->data[0]);
        ct_tensor_free(x);
        ct_tensor_free(w);
        ct_tensor_free(y);
        ct_tensor_free(L);
        
        float ra[] = {1, 2, 3, 4, 5, 6};
        size_t sa[] = {2, 3};
        float rb[] = {10, 20, 30};
        size_t sb[] = {3};
        CTTensor* a = ct_tensor_from_data(ra, 2, sa);
        CTTensor* b = ct_tensor_from_data(rb, 1, sb);
        a->requires_grad = 1;
        b->requires_grad = 1;
        L = ct_tensor_sub(a, b);
        ct_tensor_backward(L);
        printf("a.grad = "); ct_tensor_print(a->grad);
        printf("b.grad = "); ct_tensor_print(b->grad);
        ct_tensor_free(a);
        ct_tensor_free(b);
        ct_tensor_free(L);
    }

    printf("done\n");
    return 0;
}