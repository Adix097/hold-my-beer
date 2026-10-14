#include "ctorch.h"

int main(void) {
    size_t shape[] = {2, 2};
    CTTensor* a = ct_tensor_ones(2, shape);
    printf("fresh: refcount = %d\n", a->refcount);

    CTTensor* b = ct_tensor_retain(a); /* b and a are the SAME header */
    printf("after retain: refcount = %d, same pointer: %s\n", a->refcount, a == b ? "yes" : "no");

    ct_tensor_free(b);
    printf("after one free: refcount = %d (still alive)\n", a->refcount);

    CTTensor* v = ct_tensor_transpose(a, 0, 1); /* new header, shares storage */
    printf("view refcount = %d, storage refcount = %d\n", v->refcount, a->storage->refcount);

    ct_tensor_free(a); /* last tensor ref: header freed, storage lives on for v */
    printf("freed a, view still reads: %.1f\n", v->storage->data[0]);

    ct_tensor_free(v);
    printf("done\n");
    return 0;
}