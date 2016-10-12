#include <assert.h>
#include <fcntl.h>    /* For O_* constants */
#include <getopt.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "utstring.h"
#include "probabilistic.h"


double LOG_PROB = 0.0;

/**
 * Dummy "predict" implementation just writes to STDOUT immediately
 *
 */ 
void predict(const char *format, ...) {
    UT_string *contents;
	utstring_new(contents);
    va_list args;
    va_start(args, format);
    utstring_printf_va(contents, format, args);
    va_end(args);
    printf("%s", utstring_body(contents));
    utstring_free(contents);
}

/**
 * Weighting function accumulates the log-probability of the observes
 *
 */
void weight_trace(const double ln_p, const bool synchronize) {
    LOG_PROB += ln_p;
}


/** 
 * Dummy "infer" which just runs the program once
 *
 */
int infer(int (*f)(int, char**), int argc, char **argv) {
    erp_rng_init();
    int retval = __program(argc, argv);
    printf("trace_weight,%f\n", LOG_PROB);
    return retval;
}


/**
 * No args to parse
 *
 */
void parse_args(int argc, char **argv) { }

