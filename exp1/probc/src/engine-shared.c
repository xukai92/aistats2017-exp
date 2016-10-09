#include <assert.h>
#include <fcntl.h>    /* For O_* constants */
// #include <getopt.h>
// #include <stdarg.h>
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

#include "probabilistic.h"
#include "engine-shared.h"



// Name of file pointer for shared memory object
static char SHM_FILE[256];






/**
 * Special printf function "predict", for named doubles
 *
 */
void predict_float(const char *name, const double value) {
    predict("%s,%f\n", name, value);
}

void predict_int(const char *name, const int value) {
    predict("%s,%d\n", name, value);
}

void predict_chars(const char *name, const char *chars) {
    predict("%s,%s\n", name, chars);
}



double log_sum_exp(double *log_values, int count) {
    double max_log_value = log_values[0];
    for (int i=1; i<count; i++) {
        if (max_log_value < log_values[i]) {
            max_log_value = log_values[i];
        }
    }
    double return_val = 0;
    for (int i=0; i<count; i++) {
        return_val += exp(log_values[i] - max_log_value);
    }
    return_val = max_log_value + log(return_val);
    return return_val;
}


/**
 * Initialize a mutex, cond pair into shared memory for synchronizing across processes
 *
 */
void init_shared_mutex(pthread_mutex_t *mutex, pthread_cond_t *cond) {
    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(mutex, &mutex_attr);

    if (cond != NULL) {
        pthread_condattr_t cond_attr;
        pthread_condattr_init(&cond_attr);
        pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
        pthread_cond_init(cond, &cond_attr);
    }
}



void print_walltime(pthread_mutex_t *mutex, int iteration_count, struct timeval *start_time) {
    pthread_mutex_lock(mutex);
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    long seconds_elapsed = current_time.tv_sec - start_time->tv_sec;
    int usec_elapsed = current_time.tv_usec - start_time->tv_usec;
    if (usec_elapsed < 0) {
        usec_elapsed += 1e6;
        seconds_elapsed--;
    }
    fprintf(stdout, "time_elapsed,%ld.%06d,,%d\n", seconds_elapsed, usec_elapsed, iteration_count-1);
    fflush(stdout);
    pthread_mutex_unlock(mutex);
}


/**
 * In all implementations, observe is just an alias for weight_trace
 *
 */
void observe(const double ln_p) {
    weight_trace(ln_p, true);
}


/**
 * Gobble up excess children
 *
 */
void cleanup_children(int num_children_to_eat, int *const total_children) {
    debug_print(4,"Preparing to gobble up %d children of process %d\n", num_children_to_eat, getpid());
    while (num_children_to_eat > 0) {
        int status = 0;
        pid_t terminated_pid = wait(&status);
        if (status != 0) {
            perror("wait");
            debug_print(4,"[ERROR] unable to eat child process of pid %d (terminated pid = %d)\n", getpid(), terminated_pid);
        }
        num_children_to_eat--;
        *total_children = *total_children - 1;
        debug_print(4,"Child process %d->%d terminated (%d remaining)\n", getpid(), terminated_pid, *total_children);
    }
}


/**
 * Gobble up excess children, non-blocking version (only eats already-dead babies)
 *
 */
void cleanup_completed_children(int *const total_children) {
    debug_print(4,"Preparing to gobble up already-exited children of process %d\n", getpid());
    while (*total_children > 0) {
        int status = 0;
        pid_t terminated_pid = waitpid(-1, &status, WNOHANG);
        if (status != 0) {
            perror("wait");
            debug_print(4,"[ERROR] unable to eat child process of pid %d (terminated pid = %d)\n", getpid(), terminated_pid);
        }
        if (terminated_pid > 0) {
            // A positive return value indicates we actually collected a terminated child process
            *total_children = *total_children - 1;
        } else if (terminated_pid == 0) {
            // No child particles have terminated; return
            break;
        } else {
            // There's been an error.
            // TODO handle
        }
    }
}


/**
 * Flush predict buffer
 *
 */
void flush_output(pthread_mutex_t *mutex, UT_string *buffer) {
    pthread_mutex_lock(mutex);
    int stdout_copy = dup(STDOUT_FILENO);
    FILE* out = fdopen(stdout_copy, "w");
    fprintf(out, "%s", utstring_body(buffer));
    fflush(out);
    // TODO for some reason, calling fclose() here can freeze, on linux.
    //fclose(out);
    pthread_mutex_unlock(mutex);
}


/**
 * Helper function for allocating shared memory blocks with mmap.
 *
 */
void *shared_memory_alloc(int mem_size) {
    int descriptor = -1;
    descriptor = shm_open(SHM_FILE, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    ftruncate(descriptor, mem_size);
    void * object = mmap(NULL, mem_size,
            PROT_WRITE | PROT_READ, MAP_SHARED,
            descriptor, 0 );
    if (MAP_FAILED == object) {
        perror("mmap");
    }
    assert(MAP_FAILED != object);
    shm_unlink(SHM_FILE);
    return object;
}


/**
 * Program execution wrapper
 *
 */
int program_execution_wrapper(int argc, char **argv) {
    // clock_t start, end;
    // FILE *fp;
    // start = clock();
    parse_args(argc, argv);

    char *fixed_argv[argc];
    for (int argi = 0; argi < argc; argi++) {
        fixed_argv[argi] = argv[argi];
    }

    // Check if "--" stopped option conversion.
    for (int argi = 0; argi < argc; argi++) {
        if (strcmp(fixed_argv[argi], "--") == 0) {
            // If so, overwrite argv w/ right half of command line.
            argv = &(fixed_argv[argi]);
            argv[0] = fixed_argv[0];
            argc = argc - argi;
            break;
        }
    }

    // Set SHM_FILE based on executed path
    strcpy(SHM_FILE, argv[0]);
    assert(SHM_FILE[0] != '\0');
    SHM_FILE[0] = '/';
    int f_ix = 1;
    while (SHM_FILE[f_ix] != '\0') {
        if (SHM_FILE[f_ix] == '/') {
            SHM_FILE[f_ix] = '_';
        }
        f_ix++;
    }

    // TODO NOTE shouldn't have to unlink this, except after a crash.
    shm_unlink(SHM_FILE);

    // end = clock();
    // printf("%lu", end-start);
    return infer(&__program, argc, argv);
}
