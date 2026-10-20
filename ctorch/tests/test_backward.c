#include "ctorch.h"
#include <math.h>

static CTTensor* scalar(float value) {
    size_t shape[] = {1};

    CTTensor* tensor = ct_tensor_from_data(&value, 1, shape);
    tensor->requires_grad = 1;

    return tensor;
}

typedef CTTensor* (*UnaryFn)(const CTTensor*);

static void check(const char* name, UnaryFn function, float value) {
    CTTensor* input = scalar(value);
    CTTensor* output = function(input);

    ct_tensor_backward(output);

    float analytic = input->grad->storage->data[0];

    float eps = 1e-2f;

    CTTensor* input_hi = scalar(value + eps);
    CTTensor* input_lo = scalar(value - eps);

    CTTensor* output_hi = function(input_hi);
    CTTensor* output_lo = function(input_lo);

    float numeric = (output_hi->storage->data[0] - output_lo->storage->data[0]) / (2.0f * eps);

    printf("%-5s at %.1f: backward = %.4f, numeric = %.4f\n", name, value, analytic, numeric);

    ct_tensor_free(input);
    ct_tensor_free(output);

    ct_tensor_free(input_hi);
    ct_tensor_free(input_lo);
    ct_tensor_free(output_hi);
    ct_tensor_free(output_lo);
}

int main(void) {
    CTTensor* x1 = scalar(2.0f);
    CTTensor* w1 = scalar(3.0f);

    CTTensor* y1 = ct_tensor_mul(x1, w1);
    CTTensor* loss1 = ct_tensor_add(y1, x1);

    ct_tensor_backward(loss1);

    printf("\n=== Multiplication + Addition ===\n");
    printf("x.grad = %.1f (expect 4)\n", x1->grad->storage->data[0]);
    printf("w.grad = %.1f (expect 2)\n", w1->grad->storage->data[0]);
    printf("y.grad = %.1f (expect 1)\n", y1->grad->storage->data[0]);

    ct_tensor_free(x1);
    ct_tensor_free(w1);
    ct_tensor_free(y1);
    ct_tensor_free(loss1);

    float a_data[] = {1, 2, 3, 4, 5, 6};
    size_t a_shape[] = {2, 3};

    float b_data[] = {10, 20, 30};
    size_t b_shape[] = {3};

    CTTensor* a1 = ct_tensor_from_data(a_data, 2, a_shape);
    CTTensor* b1 = ct_tensor_from_data(b_data, 1, b_shape);

    a1->requires_grad = 1;
    b1->requires_grad = 1;

    CTTensor* product1 = ct_tensor_mul(a1, b1);

    ct_tensor_backward(product1);

    printf("\n=== Broadcasting Multiplication ===\n");
    printf("a.grad = ");
    ct_tensor_print(a1->grad);

    printf("b.grad = ");
    ct_tensor_print(b1->grad);

    ct_tensor_free(a1);
    ct_tensor_free(b1);
    ct_tensor_free(product1);

    CTTensor* x2 = scalar(5.0f);
    CTTensor* w2 = scalar(2.0f);

    CTTensor* loss2 = ct_tensor_sub(x2, w2);

    ct_tensor_backward(loss2);

    printf("\n=== Subtraction ===\n");
    printf("x.grad = %.4f (expect 1)\n", x2->grad->storage->data[0]);
    printf("w.grad = %.4f (expect -1)\n", w2->grad->storage->data[0]);

    ct_tensor_free(x2);
    ct_tensor_free(w2);
    ct_tensor_free(loss2);

    CTTensor* x3 = scalar(6.0f);
    CTTensor* w3 = scalar(3.0f);

    CTTensor* loss3 = ct_tensor_div(x3, w3);

    ct_tensor_backward(loss3);

    printf("\n=== Division ===\n");
    printf("x.grad = %.4f (expect 0.3333)\n", x3->grad->storage->data[0]);
    printf("w.grad = %.4f (expect -0.6667)\n", w3->grad->storage->data[0]);

    ct_tensor_free(x3);
    ct_tensor_free(w3);
    ct_tensor_free(loss3);

    CTTensor* x4 = scalar(2.0f);
    CTTensor* w4 = scalar(3.0f);

    CTTensor* product2 = ct_tensor_mul(x4, w4);
    CTTensor* loss4 = ct_tensor_neg(product2);

    ct_tensor_backward(loss4);

    printf("\n=== Negation ===\n");
    printf("x.grad = %.1f (expect -3)\n", x4->grad->storage->data[0]);
    printf("w.grad = %.1f (expect -2)\n", w4->grad->storage->data[0]);

    ct_tensor_free(x4);
    ct_tensor_free(w4);
    ct_tensor_free(product2);
    ct_tensor_free(loss4);

    float c_data[] = {1, 2, 3, 4, 5, 6};
    size_t c_shape[] = {2, 3};

    float d_data[] = {10, 20, 30};
    size_t d_shape[] = {3};

    CTTensor* c = ct_tensor_from_data(c_data, 2, c_shape);
    CTTensor* d = ct_tensor_from_data(d_data, 1, d_shape);

    c->requires_grad = 1;
    d->requires_grad = 1;

    CTTensor* loss5 = ct_tensor_sub(c, d);

    ct_tensor_backward(loss5);

    printf("\n=== Broadcasting Subtraction ===\n");
    printf("c.grad = ");
    ct_tensor_print(c->grad);

    printf("d.grad = ");
    ct_tensor_print(d->grad);

    ct_tensor_free(c);
    ct_tensor_free(d);
    ct_tensor_free(loss5);

    printf("\n=== Unary Gradient Checks ===\n");

    check("exp",  ct_tensor_exp,  1.5f);
    check("log",  ct_tensor_log,  2.0f);
    check("sqrt", ct_tensor_sqrt, 4.0f);

    check("relu", ct_tensor_relu,  3.0f);
    check("relu", ct_tensor_relu, -3.0f);

    CTTensor* x5 = scalar(1.0f);
    CTTensor* w5 = scalar(2.0f);

    CTTensor* product3 = ct_tensor_mul(x5, w5);
    CTTensor* loss6 = ct_tensor_exp(product3);

    ct_tensor_backward(loss6);

    float expected_exp2 = expf(2.0f);

    printf("\n=== Chained Autograd ===\n");
    printf("x.grad = %.4f (expect %.4f)\n", x5->grad->storage->data[0], 2.0f * expected_exp2);

    printf("w.grad = %.4f (expect %.4f)\n", w5->grad->storage->data[0], expected_exp2);

    ct_tensor_free(x5);
    ct_tensor_free(w5);
    ct_tensor_free(product3);
    ct_tensor_free(loss6);

    printf("\ndone\n");
    return 0;
}