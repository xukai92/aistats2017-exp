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
#include "engine-shared.h"

// Profiling
clock_t start, end;
FILE *fp;

// Set defaults for number of particles
static int NUM_PARTICLES = 100;

// We can print out an estimate of the marginal likelihood, as we run SMC
static bool ESTIMATE_MARGINAL_LIKELIHOOD = false;

// Tau \in [0, 1] determines the frequency of resampling
static double TAU = 0.5;

// Possibly default initial seed
static long INITIAL_SEED = -1;

// Flag to mark whether or not to record walltime
static bool TIME_EXECUTION = false;

// Flag to mark whether to output weighted or unweighted particle set
static bool WEIGHTED_OUTPUT = false;


/**
 * Struct containing global (shared) state variables
 *
 */
typedef struct {

    // Hold per-particle log-weights and number of offspring, for resampling
    double *log_weights;
    int *n_offspring;

    int current_observe;

    // Synchronization state

    // Barrier: all particles have reached an observe
    int begin_observe_counter;
    pthread_mutex_t begin_observe_mutex;
    pthread_cond_t begin_observe_cond;

    // Barrier: all particles have completed an observe
    int end_observe_counter;
    pthread_mutex_t end_observe_mutex;
    pthread_cond_t end_observe_cond;

    // Barrier: all particles completed program execution
    int exec_complete_counter;
    pthread_mutex_t exec_complete_mutex;
    pthread_cond_t exec_complete_cond;

    // Mutex: stdout lock
    pthread_mutex_t stdout_mutex;

    // Mutex: particle id
    int particle_id;
    pthread_mutex_t particle_id_mutex;

    // Marginal likelihood estimate
    double log_marginal_likelihood;

} shared_globals;

/**
 * Struct containing local state of particle
 *
 */
typedef struct {
    double log_weight;
    double log_likelihood;
    int current_observe;
    int live_offspring_count;
    UT_string *predict;
} process_locals;


static process_locals *locals;
static shared_globals *globals;


/**
 * Sample number of offspring, given particle weights
 * Basic multinomial resampling scheme
 *
 */
void multinomial_resample() {

    // Generate sampling distribution from unnormalized weights
    int s;
    double log_denominator = log_sum_exp(globals->log_weights, NUM_PARTICLES);
    double *sampling_dist = malloc(NUM_PARTICLES*sizeof(double));
    for (s=0; s<NUM_PARTICLES; s++) {
        sampling_dist[s] = exp(globals->log_weights[s] - log_denominator);
        globals->n_offspring[s] = 0;
    }

    // Draw number of offspring.
    for (s=0; s<NUM_PARTICLES; s++) {
        globals->n_offspring[discrete_rng(sampling_dist, NUM_PARTICLES)]++;
    }


#if DEBUG_LEVEL >= 2
    // print all the offspring counts (debug)
    fprintf(stderr, "[resampling %d] observe #%d\n", getpid(), locals->current_observe);
    fprintf(stderr, "P(CHILD): <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%0.4f ", sampling_dist[i]); }
    fprintf(stderr, ">\n");
    fprintf(stderr, "LOG WEIGHT: <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%0.4f ", globals->log_weights[i]); }
    fprintf(stderr, ">\n");
    fprintf(stderr, "N_OFFSPRING: <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%d ", globals->n_offspring[i]); }
    fprintf(stderr, ">\n");
#endif

    free(sampling_dist);
}


/**
 * Sample number of offspring, given particle weights
 * Residual resampling scheme
 *
 */
void residual_resample() {

    // Generate sampling distribution from unnormalized weights
    int s;
    double log_denominator = log_sum_exp(globals->log_weights, NUM_PARTICLES);
    double *sampling_dist = malloc(NUM_PARTICLES*sizeof(double));
    int remainder = NUM_PARTICLES;
    for (s=0; s<NUM_PARTICLES; s++) {
        sampling_dist[s] = exp(globals->log_weights[s] - log_denominator);
        globals->n_offspring[s] = (int)floor(NUM_PARTICLES*sampling_dist[s]);
        remainder -= globals->n_offspring[s];
    }

    // Draw number of offspring.
    for (s=0; s<remainder; s++) {
        globals->n_offspring[discrete_rng(sampling_dist, NUM_PARTICLES)]++;
    }


#if DEBUG_LEVEL >= 2
    // print all the offspring counts (debug)
    fprintf(stderr, "[resampling %d] observe #%d\n", getpid(), locals->current_observe);
    fprintf(stderr, "P(CHILD): <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%0.4f ", sampling_dist[i]); }
    fprintf(stderr, ">\n");
    fprintf(stderr, "LOG WEIGHT: <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%0.4f ", globals->log_weights[i]); }
    fprintf(stderr, ">\n");
    fprintf(stderr, "N_OFFSPRING: <");
    for (int i=0; i<NUM_PARTICLES; i++) { fprintf(stderr, "%d ", globals->n_offspring[i]); }
    fprintf(stderr, ">\n");
#endif

    free(sampling_dist);
}




/**
 * Destroy current particle: free local memory, and exit
 *
 */
void destroy_particle() {
    assert(locals->live_offspring_count == 0);
    utstring_free(locals->predict);
    _exit(0);
}


/**
 * Special printf function which writes to the output file.
 *
 */
void predict(const char *format, ...) {
    va_list args;
    va_start(args, format);
    utstring_printf_va(locals->predict, format, args);
    va_end(args);
}

/**
 * Special printf function "predict", for named doubles
 *
 */
void predict_value(const char *name, const double value) {
    // generate "predict" queries, given name and value.
    utstring_printf(locals->predict,"%s,%f\n", name, value);
}

void weight_trace(const double ln_p, const bool synchronize) {

    // Accumulate overall log-likelihood
    locals->log_likelihood += ln_p;

    // If this isn't a synchronizing observe, we accumulate log probability
    // and continue normal program execution.
    if (!synchronize) {
        locals->log_weight += ln_p;
        return;
    }

    assert(locals->current_observe == globals->current_observe);

    // We want to branch and resample on every synchronizing observe
    pthread_mutex_lock(&(globals->begin_observe_mutex));
    int particles_to_count = NUM_PARTICLES;
    int shared_globals_index = globals->begin_observe_counter;
    locals->log_weight += ln_p;
    globals->log_weights[shared_globals_index] = locals->log_weight;
    globals->begin_observe_counter += 1;
    debug_print(3, "Incrementing observe counter %d to one higher than global observe counter %d [index %d, %d]\n", locals->current_observe, globals->current_observe, shared_globals_index, getpid());
    locals->current_observe += 1;

    debug_print(4,"[OBSERVE %d, %d] #%d, %0.4f\n", locals->current_observe, getpid(), globals->begin_observe_counter, ln_p);

    // TODO check, fix

    // Wait until processes are synchronized
    debug_print(3,"[observe #%d] #%d\n", locals->current_observe, globals->begin_observe_counter);
    if (globals->begin_observe_counter >= particles_to_count) {
        debug_print(4,"%d: observed %d of %d particles, moving on\n", getpid(), globals->begin_observe_counter, particles_to_count);

        // Reset observe counters to zero
        globals->begin_observe_counter = 0;

        // current observe?
        ++(globals->current_observe);

        double ESS = 0;
        double normalization = log_sum_exp(globals->log_weights, NUM_PARTICLES);
        for (int i=0; i<NUM_PARTICLES; i++) {
            ESS += pow(exp(globals->log_weights[i] - normalization), 2);
            globals->n_offspring[i] = 1;
        }
        ESS = 1 / ESS;
        debug_print(2,"ESS at observe %d: %f\n", locals->current_observe, ESS);
        if (ESS < TAU*NUM_PARTICLES) {

            globals->log_marginal_likelihood += normalization - log(NUM_PARTICLES);

            // sample offspring counts
            multinomial_resample();
            //residual_resample();
            for (int i=0; i<NUM_PARTICLES; i++) {
                globals->log_weights[i] = 0;
            }
        }

//        int total_offspring = 0;
//        for (int i=0; i<NUM_PARTICLES; i++) {
//            total_offspring += globals->n_offspring[i];
//        }
//        assert(total_offspring == NUM_PARTICLES);
        pthread_mutex_lock(&globals->end_observe_mutex);
        globals->end_observe_counter = NUM_PARTICLES;
        for (int i=0; i<NUM_PARTICLES; i++) {
            if (globals->n_offspring[i] == 0) globals->end_observe_counter++;
        }
        pthread_mutex_unlock(&globals->end_observe_mutex);

        // Inform peer particles that synchronization for this observe is complete
        debug_print(3,"[broadcast begin_observe] observe = %d\n", locals->current_observe);
        debug_print(2,"New observe global: %d (at local: %d)\n", globals->current_observe, locals->current_observe);
        //assert(globals->end_observe_counter == 0);
        pthread_cond_broadcast(&globals->begin_observe_cond);
    } else {
        debug_print(4,"%d: observed %d of %d particles, waiting...\n", getpid(), globals->begin_observe_counter, particles_to_count);
        // This *looks* strange, but the begin_observe_counter is incremented every
        // time this function is called, and then reset to zero before broadcast().
        debug_print(3,"[wait begin_observe %d %d] observe barrier counter = %d (pid %d)\n", locals->current_observe, globals->current_observe, globals->begin_observe_counter, getpid());
        while (globals->begin_observe_counter != 0) {
        //while (locals->current_observe != globals->current_observe) {
            pthread_cond_wait(&globals->begin_observe_cond, &globals->begin_observe_mutex);
        }
    }
    pthread_mutex_unlock(&(globals->begin_observe_mutex));
    debug_print(2, "Mutex released, asserting local %d == global %d [index %d, %d]\n", locals->current_observe, globals->current_observe, shared_globals_index, getpid());
    assert(locals->current_observe == globals->current_observe);
    locals->log_weight = globals->log_weights[shared_globals_index];


    // Spawn children
    int n_offspring = globals->n_offspring[shared_globals_index];
    if (n_offspring == 0) {
        debug_print(4, "Post resample: terminating process %d (waiting %d children)\n", getpid(), locals->live_offspring_count);
        pthread_mutex_lock(&globals->end_observe_mutex);
        globals->end_observe_counter--;
        if (globals->end_observe_counter == 0) {
            pthread_cond_broadcast(&globals->end_observe_cond);
        }
        debug_print(2, "Killed particle %d, counter down to %d\n", getpid(), globals->end_observe_counter);
        pthread_mutex_unlock(&globals->end_observe_mutex);

        cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
        destroy_particle();
        assert(false); // Unreachable line of code, hopefully
    } else {
        while (n_offspring > 1) {
            unsigned long seed = gen_new_rng_seed();
            pid_t child_pid = fork();
            if (child_pid == 0) {
                set_rng_seed(seed);
                locals->live_offspring_count = 0;
                break;
            } else if (child_pid > 0) {
                n_offspring--;
                locals->live_offspring_count++;
            } else {
                debug_print(2, "ERROR WHILE FORKING %d\n", locals->current_observe);
                perror("fork");
                sleep(1);
            }
        }
    }

    pthread_mutex_lock(&globals->end_observe_mutex);
    globals->end_observe_counter--;
    debug_print(2, "%d particles remaining [index %d, %d]\n", globals->end_observe_counter, shared_globals_index, getpid());
    if (globals->end_observe_counter == 0) {
        debug_print(2,"END OF OBSERVE %d\n", globals->current_observe);
        pthread_cond_broadcast(&globals->end_observe_cond);
    } else {
        while (globals->end_observe_counter > 0) {
            pthread_cond_wait(&globals->end_observe_cond, &globals->end_observe_mutex);
        }
    }
    pthread_mutex_unlock(&globals->end_observe_mutex);
    assert(locals->current_observe == globals->current_observe);
    debug_print(2, "[index %d, %d] I am through with observe %d\n", shared_globals_index, getpid(), locals->current_observe);
}


/**
 *
 * Initialize global state
 *
 */
void init_globals() {

    // Allocate shared memory
    globals = (shared_globals *)shared_memory_alloc(sizeof(shared_globals));
    globals->log_weights = (double *)shared_memory_alloc(NUM_PARTICLES*sizeof(double));
    globals->n_offspring = (int *)shared_memory_alloc(NUM_PARTICLES*sizeof(int));

    // Initialize process locks
    init_shared_mutex(&globals->exec_complete_mutex, &globals->exec_complete_cond);
    init_shared_mutex(&globals->begin_observe_mutex, &globals->begin_observe_cond);
    init_shared_mutex(&globals->end_observe_mutex, &globals->end_observe_cond);
    init_shared_mutex(&globals->stdout_mutex, NULL);
    init_shared_mutex(&globals->particle_id_mutex, NULL);

    // Initialize globals
    globals->begin_observe_counter = 0;
    globals->exec_complete_counter = 0;
    globals->particle_id = 0;
    globals->log_marginal_likelihood = 0.0;
}


/**
 *
 * initialize engine and start inference over a supplied program
 *
 */
int infer(int (*f)(int, char**), int argc, char **argv) {

fp = fopen("/Users/kai/Turing/exps/aistats2017/exp1/probc/fork.csv", "w");

    pid_t main_pid = getpid();
    debug_print(1, "Main process pid: %d\n", main_pid);

    // Initialize random number generators
    erp_rng_init();
    if (INITIAL_SEED >= 0) set_rng_seed(INITIAL_SEED);

    // Create shared globals
    init_globals();

    // Create initial state (pre-fork)
    process_locals _locals;
    locals = &_locals;
    locals->live_offspring_count = 0;
    locals->log_likelihood = 0;
    locals->log_weight = 0;
	utstring_new(locals->predict);

    // Get memory required for struct
    int mem_size = sizeof(shared_globals) + NUM_PARTICLES*(sizeof(double) + sizeof(int));
    debug_print(1, "Shared memory size: %d bytes\n", mem_size);

    // Start timer
    struct timeval start_time;
    if (TIME_EXECUTION) {
        gettimeofday(&start_time, NULL);
        debug_print(1, "Starting timer at %ld.%06d\n", start_time.tv_sec, (int)start_time.tv_usec);
    }

    // Run SMC once

    locals->current_observe = 0;
    globals->current_observe = 0;

    globals->exec_complete_counter = 0;

    for (int i=0; i<NUM_PARTICLES; i++) {
        // We need to set each particle with a distinct random number seed
        unsigned long int seed = gen_new_rng_seed();

        // Fork and run
        start = clock();
        pid_t child_pid = fork();
        end = clock();
        if (child_pid == 0) {

            // Child process: run program
            locals->live_offspring_count = 0;
            debug_print(4,"new child rng seed: %ld\n", seed);
            set_rng_seed(seed);

            debug_print(4,"[%d -> %d]\n", main_pid, getpid());

            f(argc, argv);

            if (ESTIMATE_MARGINAL_LIKELIHOOD) {
                globals->log_marginal_likelihood += log_sum_exp(globals->log_weights, NUM_PARTICLES) - log(NUM_PARTICLES);
            }

            if (!WEIGHTED_OUTPUT) {
                observe(0); // "dummy" observe to mark end of program.

                double excess_weight = log_sum_exp(globals->log_weights, NUM_PARTICLES) - log(NUM_PARTICLES);
                if (excess_weight > 0) {
                    multinomial_resample();
                }
                flush_output(&globals->stdout_mutex, locals->predict);
            } else {

                UT_string *tmp_output;
                utstring_new(tmp_output);
                int ix_left = 0;
                int ix_right = 0;

                pthread_mutex_lock(&globals->particle_id_mutex);
                int particle_id = globals->particle_id;
                globals->particle_id++;
                while ((ix_right = utstring_find(locals->predict, ix_left, "\n", 1)) >= 0) {
                    //printf("next newline at %d\n", ix_right);
                    //printf("len: %d\n", ix_right - ix_left);
                    utstring_printf(tmp_output, "%.*s,%f,%d\n", ix_right - ix_left, &utstring_body(locals->predict)[ix_left], locals->log_weight, particle_id);
                    ix_left = ix_right + 1;
                }
                pthread_mutex_unlock(&globals->particle_id_mutex);
                flush_output(&globals->stdout_mutex, tmp_output);
                utstring_free(tmp_output);
            }

            cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);

            pthread_mutex_lock(&globals->exec_complete_mutex);
            globals->exec_complete_counter++;
            if (globals->exec_complete_counter == NUM_PARTICLES) {
                pthread_cond_broadcast(&globals->exec_complete_cond);
            }
            pthread_mutex_unlock(&globals->exec_complete_mutex);

            destroy_particle();
        } else if (child_pid < 0) {
            // Error
            perror("fork");
            utstring_free(locals->predict);
            exit(1);
        } else {
         fprintf(fp, "%lu, ", (end - start));
            locals->live_offspring_count++;
        }

        // This is the parent process
        assert(child_pid > 0);
    }

    pthread_mutex_lock(&globals->exec_complete_mutex);
    while (globals->exec_complete_counter < NUM_PARTICLES) {
        debug_print(2, "Blocking on exec complete cond in main process: %d of %d complete\n", globals->exec_complete_counter, NUM_PARTICLES);
        pthread_cond_wait(&globals->exec_complete_cond, &globals->exec_complete_mutex);
    }
    pthread_mutex_unlock(&globals->exec_complete_mutex);

    // Collect terminated child processes
    debug_print(4,"Done launching particles -- waiting for %d of them to finish\n", locals->live_offspring_count);
    cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);

    // Print out timing info
    if (TIME_EXECUTION) print_walltime(&globals->stdout_mutex, 1, &start_time);

    // Print marginal likelihood estimate
    if (ESTIMATE_MARGINAL_LIKELIHOOD) {
        pthread_mutex_lock(&globals->stdout_mutex);
        fprintf(stdout, "log_marginal_likelihood,%0.8f,,%d\n", globals->log_marginal_likelihood, NUM_PARTICLES);
        fflush(stdout);
        pthread_mutex_unlock(&globals->stdout_mutex);
    }

    utstring_free(locals->predict);
    return 0;
}


void parse_args(int argc, char **argv) {

    // Parse args
    static struct option long_options[] = {
        {"particles", required_argument, 0, 'p'},
        {"timeit", no_argument, 0, 't'},
        {"weighted", no_argument, 0, 'w'},
        {"evidence", no_argument, 0, 'e'},
        {"rng_seed", required_argument, 0, 'r'}
    };
    int c, option_index;

    while((c = getopt_long(argc, argv, "p:twer:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'p':
                NUM_PARTICLES = atoi(optarg);
                break;
            case 't':
                TIME_EXECUTION = true;
                break;
            case 'w':
                WEIGHTED_OUTPUT = true;
                break;
            case 'e':
                ESTIMATE_MARGINAL_LIKELIHOOD = true;
                break;
            case 'r':
                INITIAL_SEED = atol(optarg);
                break;
        }
    }

    debug_print(1, "Running SMC with %d particles\n", NUM_PARTICLES);
}
