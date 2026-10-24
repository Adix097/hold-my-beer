#include "ctorch.h"

static CTTensor* f_broadcast(CTTensor* const* in) {
    CTTensor* t = ct_tensor_div(in[0], in[1]);
    CTTensor* r = ct_tensor_mul(t, in[0]);
    ct_tensor_free(t);
    return r;
}

static CTTensor* f_matmul(CTTensor* const* in) {
    CTTensor* m = ct_tensor_matmul(in[0], in[1]);
    CTTensor* r = ct_tensor_sqrt(m);
    ct_tensor_free(m);
    return r;
}

static CTTensor* f_mean(CTTensor* const* in) {
    CTTensor* sq = ct_tensor_mul(in[0], in[0]);
    CTTensor* m = ct_tensor_mean(sq, 1, 0);
    CTTensor* r = ct_tensor_exp(m);
    ct_tensor_free(sq);
    ct_tensor_free(m);
    return r;
}

static void report(const char* name, float err) {
    printf("%-22s worst error = %.5f  %s\n", name, err, err < 0.01f ? "OK" : "FAIL");
}

int main(void) {
    float eps = 1e-2f;

    float da[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float db[] = {2.0f, 3.0f, 4.0f};
    size_t sa[] = {2, 3}, sb[] = {3};
    CTTensor* a = ct_tensor_from_data(da, 2, sa);
    CTTensor* b = ct_tensor_from_data(db, 1, sb);
    CTTensor* in1[] = {a, b};
    report("a / b * a (broadcast)", ct_gradcheck(f_broadcast, in1, 2, eps));

    float dx[] = {1, 2, 3, 4, 5, 6};
    float dy[] = {1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 3, 1};
    size_t sx[] = {2, 3}, sy[] = {3, 4};
    CTTensor* x = ct_tensor_from_data(dx, 2, sx);
    CTTensor* y = ct_tensor_from_data(dy, 2, sy);
    CTTensor* in2[] = {x, y};
    report("sqrt(X @ Y)", ct_gradcheck(f_matmul, in2, 2, eps));

    float dm[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    CTTensor* m = ct_tensor_from_data(dm, 2, sa);
    CTTensor* in3[] = {m};
    report("exp(mean(a*a, 1))", ct_gradcheck(f_mean, in3, 1, eps));

    ct_tensor_free(a);
    ct_tensor_free(b);
    ct_tensor_free(x);
    ct_tensor_free(y);
    ct_tensor_free(m);
    printf("done\n");
    return 0;
}