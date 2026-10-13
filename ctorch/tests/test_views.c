#include "ctorch.h"

int main(void) {
    float raw[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    CTTensor* a = ct_tensor_from_data(raw, 2, shape);
    printf("a    = "); ct_tensor_print(a);

    // transpose
    CTTensor* t = ct_tensor_transpose(a, 0, 1);
    printf("a^T  = "); ct_tensor_print(t);
    printf("shares storage: %s, refcount = %d\n", a->storage == t->storage ? "yes" : "no", a->storage->refcount);
    printf("a contiguous? %d, a^T contiguous? %d\n", ct_tensor_is_contiguous(a), ct_tensor_is_contiguous(t));

    // write through original and read through view
    uint64_t i01[] = {0, 1};
    ct_tensor_set(a, i01, 99.0f);
    uint64_t i10[] = {1, 0};
    printf("a[0,1]=99 -> a^T[1,0] = %.0f (expect 99)\n", ct_tensor_get(t, i10));

    CTTensor* tc = ct_tensor_contiguous(t);
    printf("a^T.contiguous() = "); ct_tensor_print(tc);
    printf("shares storage: %s\n", tc->storage == t->storage ? "yes" : "no");

    // reshape
    uint64_t flat[] = {6};
    CTTensor* r1 = ct_tensor_reshape(t, 1, flat);
    printf("a^T.reshape(6)    = "); ct_tensor_print(r1);

    uint64_t s32[] = {3, -1};
    CTTensor* r2 = ct_tensor_reshape(a, 2, s32);
    printf("a.reshape(3, -1)  = "); ct_tensor_print(r2);
    printf("r2 shares storage with a: %s\n", r2->storage == a->storage ? "yes" : "no");

    // slice a[:, 1:3] and select a[1]
    CTTensor* sl = ct_tensor_slice(a, 1, 1, 3, 1);
    printf("a[:, 1:3] = "); ct_tensor_print(sl);
    CTTensor* row = ct_tensor_select(a, 0, 1);
    printf("a[1]      = "); ct_tensor_print(row);

    ct_tensor_free(a); ct_tensor_free(t); ct_tensor_free(tc);
    ct_tensor_free(r1); ct_tensor_free(r2); ct_tensor_free(sl);
    ct_tensor_free(row);
    printf("done\n");
    return 0;
}