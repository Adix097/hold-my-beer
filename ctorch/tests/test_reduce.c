#include "ctorch.h"

int main(void) {
    float ra[] = {1, 2, 3, 4, 5, 6};
    size_t sa[] = {2, 3};
    CTTensor* a = ct_tensor_from_data(ra, 2, sa);

    CTTensor* s0 = ct_tensor_sum(a, 0, 0);
    CTTensor* s1 = ct_tensor_sum(a, 1, 0);
    CTTensor* s1k = ct_tensor_sum(a, 1, 1);
    CTTensor* m1 = ct_tensor_mean(a, 1, 0);
    CTTensor* mx0 = ct_tensor_max(a, 0, 0);
    CTTensor* all = ct_tensor_sum_all(a);
    printf("sum dim0       = "); ct_tensor_print(s0);
    printf("sum dim1       = "); ct_tensor_print(s1);
    printf("sum dim1 keep  = "); ct_tensor_print(s1k);
    printf("mean dim1      = "); ct_tensor_print(m1);
    printf("max dim0       = "); ct_tensor_print(mx0);
    printf("sum all        = "); ct_tensor_print(all);

    // reduction over a non-contiguous view
    CTTensor* at = ct_tensor_transpose(a, 0, 1);
    CTTensor* sat = ct_tensor_sum(at, 0, 0);
    printf("sum(a^T, dim0) = "); ct_tensor_print(sat);

    // middle dimension of a 3-d tensor: [2,3,4] -> [2,4], every value 3
    size_t s3[] = {2, 3, 4};
    CTTensor* ones = ct_tensor_ones(3, s3);
    CTTensor* so = ct_tensor_sum(ones, 1, 0);
    printf("sum(ones[2,3,4], dim1) = "); ct_tensor_print(so);

    // matmul
    float rb[] = {7, 8, 9, 10, 11, 12};
    size_t sb[] = {3, 2};
    CTTensor* b = ct_tensor_from_data(rb, 2, sb);
    CTTensor* mm = ct_tensor_matmul(a, b);
    printf("a @ b   = "); ct_tensor_print(mm);

    CTTensor* gram = ct_tensor_matmul(at, a);
    printf("a^T @ a = "); ct_tensor_print(gram);

    CTTensor* list[] = {a, s0, s1, s1k, m1, mx0, all, at, sat, ones, so, b, mm, gram};
    for (size_t i = 0; i < sizeof list / sizeof list[0]; i++) ct_tensor_free(list[i]);
    printf("done\n");
    return 0;
}