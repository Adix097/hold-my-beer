#include "ctorch.h"

int main(void) {
    float ra[] = {1, 2, 3, 4, 5, 6};
    size_t sa[] = {2, 3};
    CTTensor* a = ct_tensor_from_data(ra, 2, sa);

    CTTensor* s = ct_tensor_add(a, a);
    printf("a + a = "); ct_tensor_print(s);

    float rb[] = {10, 20, 30};
    size_t sb[] = {3};
    CTTensor* b = ct_tensor_from_data(rb, 1, sb);
    CTTensor* r = ct_tensor_add(a, b);
    printf("a + [10,20,30] = "); ct_tensor_print(r);

    float rc[] = {100, 1000};
    size_t sc[] = {2, 1};
    CTTensor* c = ct_tensor_from_data(rc, 2, sc);
    CTTensor* m = ct_tensor_mul(a, c);
    printf("a * [[100],[1000]] = "); ct_tensor_print(m);

    CTTensor* both = ct_tensor_add(b, c);
    printf("[10,20,30] + [[100],[1000]] = "); ct_tensor_print(both);

    size_t big[] = {2, 3};
    CTTensor* e = ct_tensor_expand(b, 2, big);
    printf("expand(b) = "); ct_tensor_print(e);

    CTTensor* at = ct_tensor_transpose(a, 0, 1);
    CTTensor* t2 = ct_tensor_mul(at, at);
    printf("a^T * a^T = "); ct_tensor_print(t2);

    float rn[] = {-1, 0, 2, -3};
    size_t sn[] = {2, 2};
    CTTensor* n = ct_tensor_from_data(rn, 2, sn);
    CTTensor* rl = ct_tensor_relu(n);
    CTTensor* ex = ct_tensor_exp(n);
    printf("relu = "); ct_tensor_print(rl);
    printf("exp  = "); ct_tensor_print(ex);

    CTTensor* all[] = {a, s, b, r, c, m, both, e, at, t2, n, rl, ex};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) ct_tensor_free(all[i]);
    printf("done\n");
    return 0;
}