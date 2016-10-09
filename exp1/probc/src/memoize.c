#include <stdio.h>
#include <string.h>

#include "memoize.h"


/////// basic memoization api

void memoize(mem_func *mf, void *f, size_t arg_size, size_t return_size) {
    *mf = (mem_func) { arg_size, return_size, NULL, f };
}


void mem_invoke(mem_func *mf, void *arg, void *result) {
    mem_invoke_stateful(mf, arg, result, NULL);
}

void mem_invoke_stateful(mem_func *mf, void *arg, void *result, void *state) {
    hashitem *item;
    HASH_FIND(hh, mf->argmap, arg, mf->arg_size, item);
    if (item != NULL) {
        memcpy(result, item->result, mf->return_size);
    } else {
        (*mf->fn)(arg, result, state);
        item = malloc(sizeof(hashitem));
        *item = (hashitem) { malloc(mf->arg_size), malloc(mf->return_size) };
        memcpy(item->arg, arg, mf->arg_size);
        memcpy(item->result, result, mf->return_size);
        HASH_ADD_KEYPTR(hh, mf->argmap, arg, mf->arg_size, item);
    }
}

int mem_cache_count(mem_func *mf) {
    return HASH_COUNT(mf->argmap);
}

int mem_cache_bytes(mem_func *mf) {
    return mem_cache_count(mf) * (sizeof(hashitem) + mf->arg_size + mf->return_size); 
}

void mem_clear(mem_func *mf) {
    // Free all hashtable entries
    hashitem *item, *tmp;
    HASH_ITER(hh, mf->argmap, item, tmp) {
        HASH_DEL(mf->argmap, item);
        free(item->arg);
        free(item->result);
        free(item);
    }
}
