#include "ctorch.h"
#include <math.h>

static float eval_loss(CTGradFn f, CTTensor** inputs) {
    int saved = ct_grad_enabled;
    ct_grad_enabled = 0;
    CTTensor* out = f(inputs);
    CTTensor* s = ct_tensor_sum_all(out);
    float v = s->storage->data[0];
    ct_tensor_free(out);
    ct_tensor_free(s);
    ct_grad_enabled = saved;
    return v;
}

float ct_gradcheck(CTGradFn f, CTTensor** inputs, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        CT_CHECK(inputs[i]->offset == 0 && ct_tensor_is_contiguous(inputs[i]), "gradcheck: inputs must be contiguous with offset 0");
        inputs[i]->requires_grad = 1;
    }

    CTTensor* out = f(inputs);
    ct_tensor_backward(out);
    ct_tensor_free(out);

    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        CTTensor* in = inputs[i];
        size_t numel = ct_tensor_numel(in);

        // grad should be read linearly
        CTTensor* g = in->grad ? ct_tensor_contiguous(in->grad) : NULL;

        for (size_t k = 0; k < numel; k++) {
            float orig = in->storage->data[k];
            in->storage->data[k] = orig + eps;
            float hi = eval_loss(f, inputs);
            in->storage->data[k] = orig - eps;
            float lo = eval_loss(f, inputs);
            in->storage->data[k] = orig;

            float numeric = (hi - lo) / (2.0f * eps);
            float analytic = g ? g->storage->data[g->offset + k] : 0.0f;

            float scale = fmaxf(1.0f, fmaxf(fabsf(analytic), fabsf(numeric)));
            float err = fabsf(analytic - numeric) / scale;
            if (err > worst) worst = err;
        }
        ct_tensor_free(g);
    }
    return worst;
}