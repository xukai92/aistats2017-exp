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


// Set defaults for number of particles and iterations
static int NUM_PARTICLES = 10;
static int NUM_ITERATIONS = 100;

// Possibly default initial seed
static long INITIAL_SEED = -1;

// Separate random number generator used for random choices within PMCMC
// static rk_state random_state;

// Flag to mark whether or not to record walltime each iteration
static bool TIME_ITERATION = false;


/**
 * Struct containing global (shared) state variables
 * 
 */
typedef struct {

    // Hold per-particle log-weights and number of offspring, for resampling
    double *log_weights;
    int *n_offspring;

    // Store per-particle predict buffer
    char **buffer;
    int *bufsize;

    // Evidence estimate
    double log_Z_hat;
    double log_Z_hat_prev;
    bool accept;

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
    
} shared_globals;

/**
 * Struct containing local state of particle
 *
 */
typedef struct {
    double log_weight;
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
 * This gets called by each particle, after program execution completes.
 * The process exits when this function returns.
 *
 */
void mh_step() {

    // Update shared globals (synchronized via mutex)
    pthread_mutex_lock(&(globals->exec_complete_mutex));
    int shared_globals_index = globals->exec_complete_counter;
    globals->exec_complete_counter += 1;
    debug_print(3,"%d of %d particles at end of program\n", globals->exec_complete_counter, NUM_PARTICLES);

    // Wait until processes are synchronized
    if (globals->exec_complete_counter == NUM_PARTICLES) {

        // TODO Metropolis-Hastings based on evidence ratio

        double log_ratio = globals->log_Z_hat - globals->log_Z_hat_prev;
        globals->accept = log(uniform_rng(0, 1)) < log_ratio;
        debug_print(2,"log(Z): %f -> %f\n", globals->log_Z_hat_prev, globals->log_Z_hat);
        debug_print(2,"accept ratio: %f\n", exp(log_ratio));
        debug_print(2,"accept proposal? %s\n", globals->accept ? "yes" : "no");
       
        if (globals->accept) {
            globals->log_Z_hat_prev = globals->log_Z_hat;
        }
       
        debug_print(3,"[broadcast exec_complete] %d\n", getpid());

        pthread_cond_broadcast(&globals->exec_complete_cond);
        
    } else {
        while(globals->exec_complete_counter < NUM_PARTICLES) {
            debug_print(3,"[wait retain_cond] retained counter = %d\n", globals->exec_complete_counter);
            pthread_cond_wait(&globals->exec_complete_cond, &globals->exec_complete_mutex);
        }
    }
    pthread_mutex_unlock(&globals->exec_complete_mutex);

    if (globals->accept) {
        char *newbuf = utstring_body(locals->predict);
        int bufsize = utstring_len(locals->predict)+1;
        assert(bufsize < globals->bufsize[shared_globals_index]);
        memcpy(globals->buffer[shared_globals_index], newbuf, bufsize);
    } else {
        utstring_clear(locals->predict);
        utstring_printf(locals->predict, "%s", globals->buffer[shared_globals_index]);
    }
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
        if (ESS < 0.5*NUM_PARTICLES) {

            globals->log_Z_hat += log_sum_exp(globals->log_weights, NUM_PARTICLES) - log(NUM_PARTICLES);
            debug_print(2,"[resample] estimate of log(Z) at %d: %f\n", locals->current_observe, globals->log_Z_hat);

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

    // Set print buffer
    globals->buffer = (char **)shared_memory_alloc(NUM_PARTICLES*sizeof(char*));
    globals->bufsize = (int *)shared_memory_alloc(NUM_PARTICLES*sizeof(int));
    
    int buf_init = 10240;
    for (int i=0; i<NUM_PARTICLES; i++) {
        globals->buffer[i] = (char *)shared_memory_alloc(buf_init*sizeof(char));
        globals->bufsize[i] = buf_init;
    }
    
    // Initialize process locks
    init_shared_mutex(&globals->exec_complete_mutex, &globals->exec_complete_cond);
    init_shared_mutex(&globals->begin_observe_mutex, &globals->begin_observe_cond);
    init_shared_mutex(&globals->end_observe_mutex, &globals->end_observe_cond);
    init_shared_mutex(&globals->stdout_mutex, NULL);
    
    // Initialize globals
    globals->begin_observe_counter = 0;
    globals->exec_complete_counter = 0;
}


/**
 *
 * initialize engine and start inference over a supplied program
 *
 */
int infer(int (*f)(int, char**), int argc, char **argv) {
    
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
	utstring_new(locals->predict);


    // Get memory required for struct
    int mem_size = sizeof(shared_globals) + NUM_PARTICLES*(sizeof(double) + sizeof(int));
    debug_print(1, "Shared memory size: %d bytes\n", mem_size);

    // Start timer
    struct timeval start_time;
    if (TIME_ITERATION) {
        gettimeofday(&start_time, NULL);
        debug_print(1, "Starting timer at %ld.%06d\n", start_time.tv_sec, (int)start_time.tv_usec);
    }

    globals->log_Z_hat_prev = log(0);    

    // Run conditional SMC over and over a bunch of times
    for (int iter=0; iter<NUM_ITERATIONS; iter++) {

        locals->current_observe = 0;
        globals->current_observe = 0;
        globals->log_Z_hat = 0;

#if DEBUG_LEVEL >= 3
       	debug_print(3,"\n----------\nPMCMC iteration %d\n----------\n", 1+iter);
#else
        debug_print(1, "PMCMC iteration %d of %d\n", 1+iter, NUM_ITERATIONS);
#endif

        globals->exec_complete_counter = 0;
        
        for (int i=0; i<NUM_PARTICLES; i++) {
            // We need to set each particle with a distinct random number seed
            unsigned long int seed = gen_new_rng_seed();

            // Fork and run
            pid_t child_pid = fork();
            if (child_pid == 0) {

                // Child process: run program
                locals->live_offspring_count = 0;
                debug_print(4,"new child rng seed: %ld\n", seed);
                set_rng_seed(seed);

                debug_print(4,"[%d -> %d]\n", main_pid, getpid());

                f(argc, argv);
                observe(0); // "dummy" observe to mark end of program.
                
                double excess_weight = log_sum_exp(globals->log_weights, NUM_PARTICLES) - log(NUM_PARTICLES);
                if (excess_weight > 0) {
                    globals->log_Z_hat += excess_weight;
                    multinomial_resample();
                }

                mh_step();

                flush_output(&globals->stdout_mutex, locals->predict);

                cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
                destroy_particle();
            } else if (child_pid < 0) {
                // Error
                perror("fork");
                utstring_free(locals->predict);
                exit(1);
            } else {
                locals->live_offspring_count++;
            }

            // This is the parent process
            assert(child_pid > 0);
        }
        
#if DEBUG_LEVEL > 0
        if (iter == NUM_ITERATIONS - 1) {            
            fprintf(stderr, "NOTE: that was the last pmcmc iteration.\n");
        }
#endif

        pthread_mutex_lock(&globals->exec_complete_mutex);
        while (globals->exec_complete_counter < NUM_PARTICLES) {
            debug_print(3, "Blocking on exec complete cond in main process: %d of %d complete\n", globals->exec_complete_counter, NUM_PARTICLES);
            pthread_cond_wait(&globals->exec_complete_cond, &globals->exec_complete_mutex);
        }
        pthread_mutex_unlock(&globals->exec_complete_mutex);

        // Collect terminated child processes
        debug_print(4,"Done launching particles -- waiting for %d of them to finish\n", locals->live_offspring_count);
        cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);

        // Print out per-iteration timing info
        if (TIME_ITERATION) print_walltime(&globals->stdout_mutex, iter+1, &start_time);
    }

    utstring_free(locals->predict);
    return 0;
}



void parse_args(int argc, char **argv) {

    // Parse args
    static struct option long_options[] = {
        {"particles", required_argument, 0, 'p'},
        {"iterations", required_argument, 0, 'i'},
        {"timeit", no_argument, 0, 't'},
        {"rng_seed", required_argument, 0, 'r'}
    };
    int c, option_index;

    while((c = getopt_long(argc, argv, "p:i:tr:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'p':
                NUM_PARTICLES = atoi(optarg);
                break;
            case 'i':
                NUM_ITERATIONS = atoi(optarg);
                break;
            case 't':
                TIME_ITERATION = true;
                break;
            case 'r':
                INITIAL_SEED = atol(optarg);
                break;
        }
    }

    debug_print(1, "Running %d iterations of %d particles each\n", NUM_ITERATIONS, NUM_PARTICLES);                
}

