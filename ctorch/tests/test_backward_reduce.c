#include "ctorch.h"

static CTTensor* make(const float* data, size_t ndim, const size_t* shape) {
    CTTensor* t = ct_tensor_from_data(data, ndim, shape);
    t->requires_grad = 1;
    return t;
}

int main(void) {
    float ra[] = {1, 2, 3, 4, 5, 6};
    size_t sa[] = {2, 3};

    CTTensor* a = make(ra, 2, sa);
    CTTensor* s = ct_tensor_sum(a, 1, 0);
    ct_tensor_backward(s);
    printf("sum dim1:     a.grad = "); ct_tensor_print(a->grad);

    CTTensor* a2 = make(ra, 2, sa);
    CTTensor* s2 = ct_tensor_sum(a2, 0, 1);
    ct_tensor_backward(s2);
    printf("sum dim0 keep: a.grad = "); ct_tensor_print(a2->grad);

    CTTensor* a3 = make(ra, 2, sa);
    CTTensor* m = ct_tensor_mean(a3, 1, 0);
    printf("mean graph node: %s (expect mean)\n", m->grad_fn->name);
    ct_tensor_backward(m);
    printf("mean dim1:    a.grad = "); ct_tensor_print(a3->grad);

    CTTensor* a4 = make(ra, 2, sa);
    CTTensor* sq = ct_tensor_mul(a4, a4);
    CTTensor* L = ct_tensor_sum(sq, 1, 0);
    ct_tensor_backward(L);
    printf("sum(a*a):     a.grad = "); ct_tensor_print(a4->grad);

    float rv[] = {1, 2, 3};
    size_t sv[] = {3};
    CTTensor* v = make(rv, 1, sv);
    CTTensor* vs = ct_tensor_sum(v, 0, 0);
    ct_tensor_backward(vs);
    printf("sum 1-d:      v.grad = "); ct_tensor_print(v->grad);

    CTTensor* all[] = {a, s, a2, s2, a3, m, a4, sq, L, v, vs};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) ct_tensor_free(all[i]);
    printf("done\n");
    return 0;
}