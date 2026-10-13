#include "ctorch.h"
#include <string.h>

#define CT_ALIGNMENT 64

CTStorage* ct_storage_new(size_t numel) {
    CTStorage* s = malloc(sizeof(CTStorage));
    CT_CHECK(s != NULL, "out of memory");

    size_t bytes = numel*  sizeof(float);
    size_t alloc = ((bytes + CT_ALIGNMENT - 1) / CT_ALIGNMENT)*  CT_ALIGNMENT;
    if (alloc == 0) alloc = CT_ALIGNMENT;

    s->data = aligned_alloc(CT_ALIGNMENT, alloc);
    CT_CHECK(s->data != NULL, "out of memory");
    memset(s->data, 0, alloc);

    s->numel = numel;
    s->refcount = 1;
    return s;
}

CTStorage* ct_storage_retain(CTStorage* s) {
    s->refcount++;
    return s;
}

void ct_storage_release(CTStorage* s) {
    if (s == NULL) return;
    CT_CHECK(s->refcount > 0, "invalid storage");
    
    if (--s->refcount == 0) {
        free(s->data);
        free(s);
    }
}