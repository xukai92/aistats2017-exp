#ifndef __PMCMC_SHARED__

#include <pthread.h>

#include "utstring.h"

// DEBUG_LEVEL. 0 = none, 1 = minimal, 2 = detailed, 3 = verbose, 4 = absurdly verbose
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 1
#endif

#define debug_print(level, fmt, ...) do {       \
    if (level <= DEBUG_LEVEL) {                 \
        int stderr_copy = dup(STDERR_FILENO);   \
        FILE* err = fdopen(stderr_copy, "w");   \
        fprintf(err, fmt, __VA_ARGS__);         \
        fflush(err); fclose(err); }             \
    } while (0) 



/**
 * Compute log(sum(exp(value))) without as much underflow
 *
 */
double log_sum_exp(double *log_values, int count);

/**
 * Helper function for allocating shared memory blocks with mmap.
 *
 */
void *shared_memory_alloc(int mem_size);

/**
 * Initialize a mutex, cond pair into shared memory for synchronizing across processes
 *
 */
void init_shared_mutex(pthread_mutex_t *mutex, pthread_cond_t *cond);


/**
 * Print wall clock time to stdout, synchronized via supplied mutex
 *
 */
void print_walltime(pthread_mutex_t *mutex, int iteration_count, struct timeval *start_time);


/**
 * Gobble up excess children.
 * First argument: number of children to eat
 * Second argument: pointer to integer representing current total child count
 *
 */
void cleanup_children(int num_children_to_eat, int *const total_children);


/**
 * Gobble up excess children; non-blocking. Cleans up any child particles
 * which have already terminated, then returns.
 *
 * Argument: pointer to integer representing current total child count
 *
 */
void cleanup_completed_children(int *const total_children);


/**
 * Flush predict buffer to stdout in a manner which is (hopefully) process- and fork-safe
 *
 */
void flush_output(pthread_mutex_t *mutex, UT_string *buffer);


#define __PMCMC_SHARED__
#endif
