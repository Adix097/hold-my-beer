#include "ctorch.h"

int main(void) {
    // test zero initialization and indexed writes
    uint64_t shape[] = {2, 3};
    CTTensor *a = ct_tensor_zeros(2, shape);
    float v = 1.0f;
    for (uint64_t i = 0; i < 2; i++) {
        for (uint64_t j = 0; j < 3; j++) {
            uint64_t idx[] = {i, j};
            ct_tensor_set(a, idx, v++);
        }
    }
    printf("a = "); ct_tensor_print(a);

    printf("raw storage: ");
    for (size_t i = 0; i < a->storage->numel; i++) {
        printf("%.0f ", a->storage->data[i]);
    }
    printf("\n");

    // test one initialization for a three-dimensional tensor
    uint64_t shape3[] = {2, 2, 2};
    CTTensor *b = ct_tensor_ones(3, shape3);
    printf("b = "); ct_tensor_print(b);

    // test tensor creation from raw data
    float raw[] = {1.5f, 2.5f, 3.5f, 4.5f};
    uint64_t shape1[] = {4};
    CTTensor *c = ct_tensor_from_data(raw, 1, shape1);
    printf("c = "); ct_tensor_print(c);

    // test indexed reads
    uint64_t idx[] = {1, 2};
    printf("a[1,2] = %.1f (expect 6.0)\n", ct_tensor_get(a, idx));

    ct_tensor_free(a);
    ct_tensor_free(b);
    ct_tensor_free(c);
    printf("done, no leaks expected\n");
    return 0;
}