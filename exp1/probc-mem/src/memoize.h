#ifndef __MEMOIZE__

/**
 * Hashitem and function wrapper typedefs
 *
 */
#include "uthash.h"

typedef struct {
    void *arg;
    void *result;
    UT_hash_handle hh;
} hashitem;

typedef struct {
    size_t arg_size;
    size_t return_size;
    hashitem *argmap;
    void (*fn)(); // (void*, void*);
} mem_func;


/**
 * Create a "memoized" version of the function at f, at *mf. The original function must 
 * be of the form:
 *
 * void f(void *arg, void *result);
 *
 * where *arg points to the argument value(s), and *result points to a location where the
 * a return value can be stored.
 *
 * arg_size and return_size are sizeof(.) for the respective types pointed to by *arg 
 * and *result.
 * 
 */
void memoize(mem_func *mf, void *f, size_t arg_size, size_t return_size);

/**
 * Invoke a memoized function. Two versions: the "normal" version, and a stateful
 * version which takes an additional trailing argument providing some external
 * state. The external state will NOT be taken into account for cache purposes.
 *
 */
void mem_invoke(mem_func *mf, void *arg, void *result);
void mem_invoke_stateful(mem_func *mf, void *arg, void *result, void *state);

/**
 * Memoized function return variables are stored in heap memory, which is released by
 * calling this function.
 *
 * Effectively, this clears the entire memoized cache.
 *
 */
void mem_clear(mem_func *mf);


/**
 * Check size of memoized cache (as element count, or as bytes)
 *
 */
int mem_cache_count(mem_func *mf);
int mem_cache_bytes(mem_func *mf);

#define __MEMOIZE__
#endif
