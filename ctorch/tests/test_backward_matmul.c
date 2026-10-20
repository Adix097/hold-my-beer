#include "ctorch.h"
#include <math.h>

static CTTensor* make(const float* data, size_t ndim, const size_t* shape) {
    CTTensor* t = ct_tensor_from_data(data, ndim, shape);
    t->requires_grad = 1;
    return t;
}

static float loss(const CTTensor* A, const CTTensor* B) {
    ct_grad_enabled = 0;
    CTTensor* C = ct_tensor_matmul(A, B);
    CTTensor* s = ct_tensor_sum_all(C);
    float v = s->storage->data[0];
    ct_tensor_free(C);
    ct_tensor_free(s);
    ct_grad_enabled = 1;
    return v;
}

int main(void) {
    float ra[] = {1, 2, 3, 4};
    float rb[] = {5, 6, 7, 8};
    size_t s22[] = {2, 2};
    CTTensor* A = make(ra, 2, s22);
    CTTensor* B = make(rb, 2, s22);
    CTTensor* C = ct_tensor_matmul(A, B);
    CTTensor* L = ct_tensor_sum(C, 0, 0); 
    CTTensor* L2 = ct_tensor_sum(L, 0, 0);
    ct_tensor_backward(L2);
    printf("dA = "); ct_tensor_print(A->grad);
    printf("dB = "); ct_tensor_print(B->grad);

    float fa[] = {0.5f, -1.0f, 2.0f, 1.5f, 0.3f, -0.7f};
    float fb[] = {1, 2, 3, 4, -1, -2, -3, -4, 0.5f, 0.25f, 2, 1};
    size_t sa[] = {2, 3}, sb[] = {3, 4};
    CTTensor* X = make(fa, 2, sa);
    CTTensor* Y = make(fb, 2, sb);
    CTTensor* Z = ct_tensor_matmul(X, Y);
    CTTensor* zs = ct_tensor_sum(Z, 0, 0);
    CTTensor* zl = ct_tensor_sum(zs, 0, 0);
    ct_tensor_backward(zl);

    float eps = 1e-2f, worst = 0.0f;
    for (size_t i = 0; i < 6; i++) {
        float orig = X->storage->data[i];
        X->storage->data[i] = orig + eps;
        float hi = loss(X, Y);
        X->storage->data[i] = orig - eps;
        float lo = loss(X, Y);
        X->storage->data[i] = orig;
        float diff = fabsf((hi - lo) / (2 * eps) - X->grad->storage->data[i]);
        if (diff > worst) worst = diff;
    }
    for (size_t i = 0; i < 12; i++) {
        float orig = Y->storage->data[i];
        Y->storage->data[i] = orig + eps;
        float hi = loss(X, Y);
        Y->storage->data[i] = orig - eps;
        float lo = loss(X, Y);
        Y->storage->data[i] = orig;
        float diff = fabsf((hi - lo) / (2 * eps) - Y->grad->storage->data[i]);
        if (diff > worst) worst = diff;
    }
    printf("finite-difference worst error over all 18 entries: %.5f (expect < 0.01)\n", worst);

    CTTensor* all[] = {A, B, C, L, L2, X, Y, Z, zs, zl};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) ct_tensor_free(all[i]);
    printf("done\n");
    return 0;
}