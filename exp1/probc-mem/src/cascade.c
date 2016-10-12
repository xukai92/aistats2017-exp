#include <assert.h>
#include <fcntl.h>    /* For O_* constants */
#include <getopt.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "utstring.h"
#include "probabilistic.h"
#include "engine-shared.h"

#define min(a, b) ((a < b) ? (a) : (b))
#define max(a, b) ((a > b) ? (a) : (b))


// Settings governing proliferation of particles
static int TARGET_EXECUTION_COUNT;    // target number of simultaneous "live" processes
static int MAX_LEAF_NODE_COUNT = 500; // maximum total number of processes

// Flag for whether to use random or deterministic resampling rule
static bool USE_RANDOM_RESAMPLING = false;

// Flag for whether to output sequential evidence estimates
static bool ESTIMATE_MARGINAL_LIKELIHOOD = false;

// Set defaults for number of particles 
static int PARTICLE_SOFT_LIMIT = 100000;

// Flag for whether to treat weight computations at separate observes as separate 
// atomic updates. This is beneficial for performance BUT requires the program to have
// a fixed number of observe statements for every execution.
// At the moment there is no downside to setting this to "true", as this implementation
// currently still requires a fixed number of observes.
static bool UPDATE_OBSERVES_PARALLEL = true;

// Flag for prerun
static bool IS_PRERUN = true;

// Possibly default initial seed
static long INITIAL_SEED = -1;

// Flag to mark whether or not to record walltime 
static bool TIME_EXECUTION = false;


/**
 * Struct containing global (shared) state variables
 * 
 */
typedef struct {

    // Count for total number of observes
    int num_observes;

    // Hold per-observe statistics
    int *num_particles;
    float *log_avg_weight;
    int *offspring_count;
    
    int *total_num_particles;
    int initial_particles;
    
    // Synchronization state

    // Atomic updates to average weights and particle counts at each observe
    pthread_mutex_t *update_observe_mutex;
    
    // Barrier: particles take turns advancing
    int execution_leaf_node_counter;
    pthread_mutex_t execution_leaf_node_mutex;
    pthread_cond_t execution_leaf_node_cond;
    
    // Process counter
    unsigned long synthetic_pid;
    pthread_mutex_t synthetic_pid_mutex;
    
    // Mutex: stdout lock
    pthread_mutex_t stdout_mutex;
    
} shared_globals;

/**
 * Struct containing local state of particle
 *
 */
typedef struct {
    double log_weight;
    double log_weight_increment;
    double log_likelihood;
    int current_observe;
    int initial_index;
    int live_offspring_count;
    int particle_pseudocount;
    UT_string *predict;
} process_locals;


static process_locals *locals;
static shared_globals *globals;


/**
 * Destroy current particle: free local memory, and exit
 *
 */
void destroy_particle() {
    assert(locals->live_offspring_count == 0);
    utstring_free(locals->predict);
    
    pthread_mutex_lock(&globals->execution_leaf_node_mutex);
    globals->initial_particles = max(globals->initial_particles, locals->initial_index+1);
    globals->execution_leaf_node_counter--;
    // Wake up another particle before terminating this one
    pthread_cond_signal(&globals->execution_leaf_node_cond);
    pthread_mutex_unlock(&globals->execution_leaf_node_mutex);

    _exit(0);
}


/**
 * Special printf function which writes to the output file.
 *
 */
void predict(const char *format, ...) {
    if (IS_PRERUN) { return; }
    va_list args;
    va_start(args, format);
    utstring_printf_va(locals->predict, format, args);
    va_end(args);
}


/**
 * Print elapsed time at end of particle
 *
 */
void print_time_elapsed(pthread_mutex_t *mutex, unsigned long synthetic_pid, struct timeval *start_time) {
    pthread_mutex_lock(mutex);
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    long seconds_elapsed = current_time.tv_sec - start_time->tv_sec;
    int usec_elapsed = current_time.tv_usec - start_time->tv_usec;
    if (usec_elapsed < 0) {
        usec_elapsed += 1e6;
        seconds_elapsed--;
    }
    fprintf(stdout, "time_elapsed,%ld.%06d,,%lu\n", seconds_elapsed, usec_elapsed, synthetic_pid);
    fflush(stdout);
    pthread_mutex_unlock(mutex);
}


void weight_trace(const double ln_p, const bool synchronize) {

    if (IS_PRERUN) {
        if (synchronize) globals->num_observes++;
        return;
    }

    // Queue index
    int queue_index = UPDATE_OBSERVES_PARALLEL ? locals->current_observe : 0;

    // Accumulate overall log-likelihood
    locals->log_likelihood += ln_p;
    locals->log_weight_increment += ln_p;
    
    // If this isn't a synchronizing observe, we accumulate log probability 
    // and continue normal program execution.
    // debug_print(1, "[observe %d] incrementing log weight by %f, from %f\n", locals->current_observe, ln_p, locals->log_weight);
    if (!synchronize) {
        return;
    }

    locals->log_weight += locals->log_weight_increment;

    // We want to branch and (potentially) resample on every synchronizing observe
    pthread_mutex_lock(&(globals->update_observe_mutex[queue_index]));

    if (ESTIMATE_MARGINAL_LIKELIHOOD) {
        globals->total_num_particles[locals->current_observe] += locals->particle_pseudocount;
    }

    int particles_launched = globals->num_particles[0];
    int particles_so_far = globals->num_particles[locals->current_observe];

    if (particles_so_far == 0) {
        // first particle
        globals->log_avg_weight[locals->current_observe] = locals->log_weight;
        globals->offspring_count[locals->current_observe] = 0;
    } else {
        // incremental update to avg
        globals->log_avg_weight[locals->current_observe] = log_sum_exp((double[2]){ log(particles_so_far) + globals->log_avg_weight[locals->current_observe], log(locals->particle_pseudocount) + locals->log_weight }, 2) - log(particles_so_far + locals->particle_pseudocount);
    }


    float new_log_weight;
    int num_offspring;

    // Time to resample!

    // Compute mean offspring count
    double ratio = exp(locals->log_weight - globals->log_avg_weight[locals->current_observe]);
    // debug_print(1, "particle %d to arrive at %d, log weight %f; log avg %f, R = %f\n", particles_so_far, locals->current_observe, locals->log_weight, globals->log_avg_weight[locals->current_observe], ratio);


    if (ratio < 1) {
        num_offspring = flip_rng(ratio);
        new_log_weight = globals->log_avg_weight[locals->current_observe];
    } else {
        if (USE_RANDOM_RESAMPLING) {
            num_offspring = floor(ratio) + flip_rng(ratio - floor(ratio));
            new_log_weight = globals->log_avg_weight[locals->current_observe];
        } else {
            if (globals->offspring_count[locals->current_observe] > min(particles_launched, particles_so_far)) {
                num_offspring = floor(ratio);
            } else {
                num_offspring = ceil(ratio);
            }
            new_log_weight = locals->log_weight - log(num_offspring);
        }
    }
    
    if (locals->current_observe+1 == globals->num_observes) {
        // no point in multiple children for final observe
        num_offspring = 1;
      
        // also, for the final observe, collapse all our pseudo-observations
        locals->log_weight += log(locals->particle_pseudocount);
        locals->particle_pseudocount = 1;
        new_log_weight = locals->log_weight;
    }

    //debug_print(1, "particle %d at %d will have %d children, outgoing weight %f\n", particles_so_far, locals->current_observe, num_offspring, new_log_weight);
    debug_print(4, "children,%d,%d\n", locals->current_observe, num_offspring);


    if (num_offspring > 1+globals->num_particles[locals->current_observe]) {
        debug_print(2, "This should be impossible! %d offspring from %d-th particle (at observe %d)\n", num_offspring, 1+globals->num_particles[locals->current_observe], locals->current_observe);
    }

    globals->num_particles[locals->current_observe]++;
    globals->offspring_count[locals->current_observe] += num_offspring;

    pthread_mutex_unlock(&(globals->update_observe_mutex[queue_index]));

    // debug_print(1, "[observe %d] outgoing weight: %f\n", locals->current_observe, new_log_weight);

    locals->current_observe += 1;
    locals->log_weight = new_log_weight;
    locals->log_weight_increment = 0;

    debug_print(3, "[OBSERVE %d] process %d, number of offspring %d\n", locals->current_observe-1, getpid(), num_offspring);


    // Spawn children
    if (num_offspring == 0) {
        debug_print(4, "Post resample: terminating process %d (waiting %d children)\n", getpid(), locals->live_offspring_count);
        cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
        destroy_particle();
        assert(false); // Unreachable line of code, hopefully
    } else {
        while (num_offspring > 0) {

            pthread_mutex_lock(&globals->execution_leaf_node_mutex);
            debug_print(3, "[%d] signal-and-wait: %d children left at %d\n", getpid(), num_offspring, locals->current_observe-1);
            pthread_cond_signal(&globals->execution_leaf_node_cond);
            pthread_cond_wait(&globals->execution_leaf_node_cond, &globals->execution_leaf_node_mutex);
            debug_print(3, "[%d] signal received: %d children left at %d\n", getpid(), num_offspring, locals->current_observe-1);
            
            // if we have too many leaf nodes, just run a single particle for now
            if (globals->execution_leaf_node_counter > MAX_LEAF_NODE_COUNT && num_offspring > 1) {
                debug_print(2, "[warning] discarding %d children at observe %d -> %d leaf nodes\n", num_offspring-1, locals->current_observe, globals->execution_leaf_node_counter);
                locals->particle_pseudocount *= num_offspring;
                num_offspring = 1;
            }

            if (num_offspring > 1) globals->execution_leaf_node_counter++;
            pthread_mutex_unlock(&globals->execution_leaf_node_mutex);

            // eat any completed particles before forking / continuing
            cleanup_completed_children(&locals->live_offspring_count);

            if (num_offspring == 1) break;

            unsigned long seed = gen_new_rng_seed();
            pid_t child_pid = fork();
            if (child_pid == 0) {
                set_rng_seed(seed);
                locals->live_offspring_count = 0;
                break;
            } else if (child_pid > 0) {
                num_offspring--;
                locals->live_offspring_count++;
            } else {
                debug_print(1, "ERROR WHILE FORKING at %d; leaf node count = %d\n", locals->current_observe, globals->execution_leaf_node_counter);
                perror("fork");
                cleanup_completed_children(&locals->live_offspring_count);
                pthread_mutex_lock(&globals->execution_leaf_node_mutex);
                globals->execution_leaf_node_counter--;
                pthread_cond_signal(&globals->execution_leaf_node_cond);
                pthread_mutex_unlock(&globals->execution_leaf_node_mutex);
//                 sleep(1);    // TODO Consider sleeping after "issues", letting a process "cool down", as it were
            }
        }
    }
}


/**
 *
 * Initialize global state
 *
 */
void init_globals() {

    // Allocate shared memory
    globals = (shared_globals *)shared_memory_alloc(sizeof(shared_globals));
    
    // Initialize process locks
    init_shared_mutex(&globals->execution_leaf_node_mutex, &globals->execution_leaf_node_cond);
    init_shared_mutex(&globals->stdout_mutex, NULL);
    init_shared_mutex(&globals->synthetic_pid_mutex, NULL);
        
    // Initialize globals
    globals->initial_particles = 0;
    globals->num_observes = 0;
    globals->synthetic_pid = 0;
    globals->execution_leaf_node_counter = 0;
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

    // Create shared-memory globals object
    init_globals();

    // Set initial state (pre-fork)
    process_locals _locals;
    locals = &_locals;
    locals->live_offspring_count = 0;
    locals->log_likelihood = 0;
    locals->log_weight = 0;
    locals->log_weight_increment = 0;
    locals->particle_pseudocount = 1;
	utstring_new(locals->predict);

    // Do initial prerun (at the moment, all this does is count the number of observes)
    pid_t prerun_pid = fork();
    if (prerun_pid == 0) {
        f(argc, argv);
        debug_print(1, "Logging observe count: %d\n", globals->num_observes);
        exit(0);
    } else if (prerun_pid < 0) {
        perror("fork");
        exit(1);
    } else {
        int status = 0;
        pid_t terminated_pid = wait(&status);
        assert(prerun_pid == terminated_pid);
    }

    debug_print(1, "Program has %d observe statements\n", globals->num_observes);

    // Allocate variables which depend on observe count
    globals->log_avg_weight = (float *)shared_memory_alloc((globals->num_observes+1)*sizeof(float));
    globals->num_particles = (int *)shared_memory_alloc((globals->num_observes+1)*sizeof(int));
    globals->offspring_count = (int *)shared_memory_alloc((globals->num_observes+1)*sizeof(int));
    globals->update_observe_mutex = (pthread_mutex_t *)shared_memory_alloc((1 + (UPDATE_OBSERVES_PARALLEL ? globals->num_observes : 0))*sizeof(pthread_mutex_t));

    if (ESTIMATE_MARGINAL_LIKELIHOOD) {
        globals->total_num_particles = (int *)shared_memory_alloc((globals->num_observes+1)*sizeof(int));
    }

    for (int i=0; i<=globals->num_observes; i++) {
        globals->num_particles[i] = 0;
        if (UPDATE_OBSERVES_PARALLEL || i == 0) {
            init_shared_mutex(&globals->update_observe_mutex[i], NULL);
        }
        if (ESTIMATE_MARGINAL_LIKELIHOOD) {
            globals->total_num_particles[i] = 0;
        }
    }

    // Start timer
    struct timeval start_time;
    if (TIME_EXECUTION) {
        gettimeofday(&start_time, NULL);
        debug_print(1, "Starting timer at %ld.%06d\n", start_time.tv_sec, (int)start_time.tv_usec);
    }

    // Run particle cascade
    locals->current_observe = 0;
    bool is_first_run = true;
    IS_PRERUN = false;
    int i = 0;
    while(true){ 
    
        locals->initial_index = i;
    
        // We need to set each particle with a distinct random number seed
        unsigned long int seed = gen_new_rng_seed();

        debug_print(1, "Starting new particle %d (%lu completed, %d live)\n", i+1, globals->synthetic_pid, globals->execution_leaf_node_counter);

        // Fork and run
        pid_t child_pid = fork();

        if (child_pid == 0) {
            // New leaf node after fork
            pthread_mutex_lock(&globals->execution_leaf_node_mutex);
            debug_print(4, "CHILD IS SENDING SIGNAL, %d\n", getpid());
            globals->execution_leaf_node_counter++;
            debug_print(4, "CHILD CONTINUING EXECUTION, %d, is the following one? %d\n", getpid(), globals->execution_leaf_node_counter);
            pthread_mutex_unlock(&globals->execution_leaf_node_mutex);

            // Child process: run program
            locals->live_offspring_count = 0;
            debug_print(4,"new child rng seed: %ld\n", seed);
            set_rng_seed(seed);

            debug_print(4,"[%d -> %d]\n", main_pid, getpid());
            f(argc, argv);
            debug_print(4,"[%d -> END]\n", getpid());

            UT_string *tmp_output;
            utstring_new(tmp_output);
            int ix_left = 0;
            int ix_right = 0;
            
            pthread_mutex_lock(&globals->synthetic_pid_mutex);
            unsigned long synthetic_pid = globals->synthetic_pid++;
            
            double final_particle_weight = locals->log_weight; // + log(locals->particle_pseudocount);
            
            while ((ix_right = utstring_find(locals->predict, ix_left, "\n", 1)) >= 0) {
                utstring_printf(tmp_output, "%.*s,%f,%ld\n", ix_right - ix_left, &utstring_body(locals->predict)[ix_left], final_particle_weight, synthetic_pid);
                ix_left = ix_right + 1;
            }

            flush_output(&globals->stdout_mutex, tmp_output);
            utstring_free(tmp_output);

            // Print out timing info
            if (TIME_EXECUTION) print_time_elapsed(&globals->stdout_mutex, synthetic_pid, &start_time);

            // Print out marginal likelihood estimate
            if (ESTIMATE_MARGINAL_LIKELIHOOD) {
                pthread_mutex_lock(&globals->stdout_mutex);
                globals->initial_particles = max(globals->initial_particles, locals->initial_index+1);
                fprintf(stdout, "initial_particles,%d,,%lu\n", globals->initial_particles, synthetic_pid);
                fprintf(stdout, "log_marginal_likelihood,%0.10f,,%lu\n", globals->log_avg_weight[globals->num_observes-1] + log(globals->total_num_particles[globals->num_observes-1]) - log(globals->initial_particles), synthetic_pid);
                fflush(stdout);
                pthread_mutex_unlock(&globals->stdout_mutex);
            }
            pthread_mutex_unlock(&globals->synthetic_pid_mutex);


            debug_print(2, "execution %lu complete, cleaning up %d children\n", synthetic_pid, locals->live_offspring_count);

            cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);

            debug_print(3, "execution %lu done with cleanup\n", synthetic_pid);

            destroy_particle();
        } else if (child_pid < 0) {
            // Error
            debug_print(1, "Fork failed in outer loop when launching particle %d; %d execution leaves remain\n", i+1, globals->execution_leaf_node_counter);
            perror("fork");

            i--;
            sleep(1);
            //utstring_free(locals->predict);
            //exit(1);
        } else {
            locals->live_offspring_count++;
            
            if (!is_first_run || globals->execution_leaf_node_counter > TARGET_EXECUTION_COUNT) {

                is_first_run = false;

                // Wait for signal to continue outer-level spawn loop
                pthread_mutex_lock(&globals->execution_leaf_node_mutex);
                debug_print(4, "MAIN LOOP (%d) IS SENDING SIGNAL TO PROCESS QUEUE LENGTH %d\n", getpid(), globals->execution_leaf_node_counter);
                pthread_cond_signal(&globals->execution_leaf_node_cond);
                debug_print(4, "POST-SIGNAL, %d PROCESSES IN QUEUE\n", globals->execution_leaf_node_counter);
                while (true) {
                    cleanup_completed_children(&locals->live_offspring_count);

                    // The main loop should not wait if nothing else is currently executing
                    debug_print(4, "MAIN LOOP TO RE-ENTER QUEUE: %d PROCESSES IN QUEUE, %d LIVE CHILD PARTICLES\n", globals->execution_leaf_node_counter, locals->live_offspring_count);
                    if (globals->execution_leaf_node_counter > 0) {
                        pthread_cond_wait(&globals->execution_leaf_node_cond, &globals->execution_leaf_node_mutex);
                    }
                    if (locals->live_offspring_count < MAX_LEAF_NODE_COUNT) {
                        debug_print(2, "Safe to launch a new particle: %d still running, leaf node count %d.\n", locals->live_offspring_count, globals->execution_leaf_node_counter);
                        break;
                    } else {
                        debug_print(2, "FAILED to launch a new particle: %d still running, leaf node count %d.\n", locals->live_offspring_count, globals->execution_leaf_node_counter);
                        // signal someone else instead
                        pthread_cond_signal(&globals->execution_leaf_node_cond);
                    }

                }
                debug_print(4, "MAIN LOOP CONTINUING EXECUTION, %d\n", getpid());
                pthread_mutex_unlock(&globals->execution_leaf_node_mutex);

            } else {
                cleanup_completed_children(&locals->live_offspring_count);
            }
        }

        // This is the parent process
        assert(child_pid != 0);
        
        // Check if we're done spawning, before continuing loop
        pthread_mutex_lock(&globals->synthetic_pid_mutex);
        bool soft_limit_hit = (globals->synthetic_pid >= PARTICLE_SOFT_LIMIT);
        pthread_mutex_unlock(&globals->synthetic_pid_mutex);
        if (soft_limit_hit) break;
        
        // Keep a stupid counter around, for now; used in logging (TODO lets not bother, maybe)
        i++;
    }
    
    // Signal any remaining execution leaves to finish
    pthread_mutex_lock(&globals->execution_leaf_node_mutex);
    debug_print(3,"Pre-cleanup; main thread complete, leaf node counter at %d\n", globals->execution_leaf_node_counter);
    debug_print(1,"Done launching particles -- waiting for %d of them to finish, with %d total leaf nodes\n", locals->live_offspring_count, globals->execution_leaf_node_counter);
    int last_offspring_count = locals->live_offspring_count;
    while (globals->execution_leaf_node_counter > 0) {
        pthread_cond_signal(&globals->execution_leaf_node_cond);
        pthread_cond_wait(&globals->execution_leaf_node_cond, &globals->execution_leaf_node_mutex);
        cleanup_completed_children(&locals->live_offspring_count);
        if (locals->live_offspring_count < last_offspring_count) {
            debug_print(1,"Waiting on %d initial particles (%lu completed, %d live)\n", locals->live_offspring_count, globals->synthetic_pid, globals->execution_leaf_node_counter);
            last_offspring_count = locals->live_offspring_count;
        }
    }
    pthread_mutex_unlock(&globals->execution_leaf_node_mutex);

    // Collect terminated child processes
    cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
    debug_print(3,"Post-cleanup; main thread complete, leaf node counter at %d\n", globals->execution_leaf_node_counter);
    debug_print(1,"Summary: total of %lu paths completed, from %d initializations\n", globals->synthetic_pid, i+1);

    utstring_free(locals->predict);
    
    return 0;
}


void parse_args(int argc, char **argv) {

    // Parse args
    static struct option long_options[] = {
        {"particles", required_argument, 0, 'p'},
        {"timeit", no_argument, 0, 't'},
        {"evidence", no_argument, 0, 'e'},
        {"rng_seed", required_argument, 0, 'r'},
        {"process_cap", required_argument, 0, 'c'},
    };
    int c, option_index;
    
    while((c = getopt_long(argc, argv, "p:ter:c:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'p':
                PARTICLE_SOFT_LIMIT = atoi(optarg);
                break;
            case 't':
                TIME_EXECUTION = true;
                break;
            case 'e':
                ESTIMATE_MARGINAL_LIKELIHOOD = true;
                break;
            case 'r':
                INITIAL_SEED = atol(optarg);
                break;
            case 'c':
                MAX_LEAF_NODE_COUNT = atoi(optarg);
                break;
        }
    }
    TARGET_EXECUTION_COUNT = (int)(0.5 * MAX_LEAF_NODE_COUNT);


    debug_print(1, "Running cascade, targetting %d total particles\n", PARTICLE_SOFT_LIMIT);

}
