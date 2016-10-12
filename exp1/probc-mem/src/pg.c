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
static int NUM_OBSERVES = 0;
static int NUM_PARTICLES = 10;
static int NUM_ITERATIONS = 100;

// Possibly default initial seed
static long INITIAL_SEED = -1;

// Separate random number generator used for random choices within PMCMC
// static rk_state random_state;

// Flag to mark whether or not to record walltime each iteration
static bool TIME_ITERATION = false;

// Flag for prerun
static bool IS_PRERUN = true;


/**
 * Struct containing per-observation (global, shared) retained particle state
 *
 */
typedef struct {
    // Retained particle: pid, and unnormalized log weight
    pid_t retained_pid;
    double retained_ln_p;

    // Sync condition: signal when it is time for this process to branch children
    bool branch_flag;
    pthread_mutex_t branch_mutex;
    pthread_cond_t branch_cond;    
} retained_particle;

/**
 * Struct containing global (shared) state variables
 * 
 */
typedef struct {
    // Flag: are we running conditional SMC?
    bool has_retained_particle;

    // Temporary variable used to select retained particle
    int next_to_retain;

    // Hold per-particle log-weights and number of offspring, for resampling
    double *log_weights;
    int *n_offspring;

    // Retained particle trace
    retained_particle *retained;

    // Synchronization state

    // Barrier: all particles have reached an observe
    int begin_observe_counter;
    pthread_mutex_t begin_observe_mutex;
    pthread_cond_t begin_observe_cond;
    
    // Condition: retained particle has been set
    bool is_retained_particle_set;
    pthread_mutex_t retained_particle_set_mutex;
    pthread_cond_t retained_particle_set_cond;

    // Barrier: all particles have finished handling observe
    int end_observe_counter;
    pthread_mutex_t end_observe_mutex;
    pthread_cond_t end_observe_cond;
    
    // Barrier: all particles completed program execution
    int exec_complete_counter;
    pthread_mutex_t exec_complete_mutex;
    pthread_cond_t exec_complete_cond;
    
    // Barrier: all observations have retained a particle
    int retain_complete_counter;
    pthread_mutex_t retain_complete_mutex;
    pthread_cond_t retain_complete_cond;
    
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
    pid_t *pid_trace;
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
    // If this is a conditional SMC run, we choose NUM_PARTICLES - 1 here.
    int offspring_to_sample = NUM_PARTICLES - ((globals->has_retained_particle) ? 1 : 0);
    for (s=0; s<offspring_to_sample; s++) {
        globals->n_offspring[discrete_rng(sampling_dist, NUM_PARTICLES)]++;
    }
    
    // If this is conditional SMC, increase retained particle offspring count by 1.
    if (globals->has_retained_particle) {
        globals->n_offspring[NUM_PARTICLES-1]++;
    }

#if DEBUG_LEVEL >= 2
    // print all the offspring counts (debug)
    fprintf(stderr, "[resampling %d] pmcmc iteration ??, observe #%d\n", getpid(), locals->current_observe);
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

    if (globals->has_retained_particle && (globals->n_offspring[NUM_PARTICLES-1] == 0)) {
        assert(remainder > 0);
        globals->n_offspring[NUM_PARTICLES-1] = 1;
        remainder--;
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
    free(locals->pid_trace);
    utstring_free(locals->predict);
    _exit(0);
}


/**
 * Main control loop.
 * Returning from this function will start a branch.
 * The original process which calls this should never return.
 *
 * When called initially, 
 * (1) kick off however many children need spawned.
 * (2) wait for a "retain" signal.
 * (3) if this particle is retained, wait for a branch signal. if not, exit.
 * (4) when we get the branch signal, go back to (1).
 *
 */
void retain_branch_loop(int children_to_spawn) {
    // bool is_first_run = true;
    pid_t parent_pid = getpid();
    while (true) {
    
//         int target_children = children_to_spawn;
        assert(locals->live_offspring_count <= 1);
    
        // If there are babies to make, go make them
        debug_print(4,"Particle %d at observe %d is going to branch %d NEW children and wait to see if it is retained\n", getpid(), locals->current_observe, children_to_spawn); 
        while (children_to_spawn > 0) {
            unsigned long int seed = gen_new_rng_seed();
            pid_t child_pid = fork();
            if (child_pid == 0) {
                // New child. Update offspring, observe index, pid trace; then continue execution
                set_rng_seed(seed);
                debug_print(4,"new child rng seed: %ld\n", seed);
                debug_print(4,"[%d -> %d]\n", parent_pid, getpid());
                locals->live_offspring_count = 0;
                locals->current_observe++;
                locals->pid_trace[locals->current_observe] = getpid();
                return;
            } else if (child_pid > 0) {
                // Parent (control) process.
                children_to_spawn--;              
                locals->live_offspring_count++;
            } else {
                // If there was an error, it's probably because the process table is full.
                // Wait a second, then retry.
                perror("fork");
                sleep(1);
            }
        }
        
        debug_print(4,"Okay: %d now has %d children\n", getpid(), locals->live_offspring_count);

        // After all the children have been made, wait until we get a "retain" signal
        pthread_mutex_lock(&(globals->retained_particle_set_mutex));

        // Count how many observes are complete; if they all are, let the other particles
        // know it is time to move to the next observe.
        pthread_mutex_lock(&globals->end_observe_mutex);
        globals->end_observe_counter++;
        debug_print(3,"[end_observe] counter = %d (wait until %d)\n", globals->end_observe_counter, NUM_PARTICLES);
        if(globals->end_observe_counter == NUM_PARTICLES) {
            pthread_cond_broadcast(&globals->end_observe_cond);
        }
        pthread_mutex_unlock(&globals->end_observe_mutex);
        
        
        while (!globals->is_retained_particle_set) {
            debug_print(3,"[wait retained_particle_set_cond] observe %d, pid %d\n", locals->current_observe, getpid());
            pthread_cond_wait(&(globals->retained_particle_set_cond), &(globals->retained_particle_set_mutex));
        }
        pthread_mutex_unlock(&(globals->retained_particle_set_mutex));	

        // If this is not the retained particle, exit.
        bool is_retained = (getpid() == globals->retained[locals->current_observe].retained_pid);
        debug_print(4,"observe %d, pid %d; retaining %d. Is retained? %d\n", locals->current_observe, getpid(), globals->retained[locals->current_observe].retained_pid, is_retained);
        if (!is_retained) { 
            // Not retained? gobble up ALL children, and exit
            cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
            destroy_particle();
        } 

        // This IS the retained particle.
        // Gobble up ALL BUT ONE child, continue
        pthread_mutex_lock(&(globals->retained[locals->current_observe].branch_mutex));
        debug_print(3,"[retained particle %d] eating %d children\n", getpid(), locals->live_offspring_count-1);
        cleanup_children(locals->live_offspring_count-1, &locals->live_offspring_count);


        // Store log_weight of retained particle
        double retained_weight = locals->log_weight;
        debug_print(4,"Updated retained log weight at observe %d: %0.4f\n", locals->current_observe, retained_weight);
        globals->retained[locals->current_observe].retained_ln_p = retained_weight;

        // Keep track of whether we have finished retaining the entire trace
        pthread_mutex_lock(&globals->retain_complete_mutex);
        globals->retain_complete_counter++;
        if (globals->retain_complete_counter == NUM_OBSERVES) {
            debug_print(3,"[broadcast retain_complete] %d\n", globals->retain_complete_counter);
            pthread_cond_broadcast(&globals->retain_complete_cond);
        }
        pthread_mutex_unlock(&globals->retain_complete_mutex);

        
        // Wait for a "branch" signal
        debug_print(3,"[wait for branch %d] %d live children\n", getpid(), locals->live_offspring_count);
        globals->retained[locals->current_observe].branch_flag = false;
        while (!globals->retained[locals->current_observe].branch_flag) {
            debug_print(3,"[wait retained[%d].branch_cond] %d\n", locals->current_observe, getpid());
            pthread_cond_wait(&(globals->retained[locals->current_observe].branch_cond), &(globals->retained[locals->current_observe].branch_mutex));
        }
        // Time to branch. Get number of children to spawn
        children_to_spawn = globals->n_offspring[NUM_PARTICLES-1] - 1;
        pthread_mutex_unlock(&(globals->retained[locals->current_observe].branch_mutex));	

        // If the number of children to spawn is NEGATIVE, it means it is time to
        // clear the retained particle, and exit.
        assert(locals->live_offspring_count == 1);
        if (children_to_spawn < 0) {
            debug_print(4,"Removing retained particle node %d (currently has %d children)\n", getpid(), locals->live_offspring_count);
            cleanup_children(locals->live_offspring_count, &locals->live_offspring_count);
            destroy_particle();
        }
        
        // Re-print retained particle PREDICT directives at end of execution trace.
        if (locals->current_observe == NUM_OBSERVES-1) {
            assert(globals->has_retained_particle);
            flush_output(&globals->stdout_mutex, locals->predict);
        }
    }
}

/**
 * This gets called by each particle, after program execution completes.
 * The process exits when this function returns.
 *
 */
void set_retained_particle() {

    // Update shared globals (synchronized via mutex)
    pthread_mutex_lock(&(globals->exec_complete_mutex));
    int shared_globals_index = globals->exec_complete_counter;
    globals->exec_complete_counter += 1;
    debug_print(3,"%d of %d particles at end of program (+%d reprint)\n", globals->exec_complete_counter, NUM_PARTICLES, globals->has_retained_particle);

    // Wait until processes are synchronized
    if (globals->exec_complete_counter + globals->has_retained_particle >= NUM_PARTICLES) {

        globals->next_to_retain = uniform_discrete_rng(NUM_PARTICLES);
       
        debug_print(3,"[broadcast retain_cond] retained particle index = %d\n", globals->next_to_retain);

        pthread_cond_broadcast(&globals->exec_complete_cond);
        //debug_print(4,"End of iteration; retaining %d\n", globals->next_to_retain);
            
        // If we're retaining the previously retained particle, handle that separately.
        // We don't need to update the PID trace; we just need to inform the stored
        // retained particle state that all the observes are complete.
        if (globals->has_retained_particle && globals->next_to_retain == NUM_PARTICLES-1) {
            pthread_mutex_lock(&(globals->retained_particle_set_mutex));
            globals->is_retained_particle_set = true;
            debug_print(3,"[broadcast retained_particle_set_cond] (retained particle %d) \n", globals->next_to_retain);
            pthread_cond_broadcast(&(globals->retained_particle_set_cond));
            pthread_mutex_unlock(&(globals->retained_particle_set_mutex));        
        }
        
        // Set static property: from now on, we are now running conditional SMC
        globals->has_retained_particle = true;
        
    } else {
        while(globals->exec_complete_counter + globals->has_retained_particle < NUM_PARTICLES) {
            debug_print(3,"[wait retain_cond] retained counter = %d\n", globals->exec_complete_counter);
            pthread_cond_wait(&globals->exec_complete_cond, &globals->exec_complete_mutex);
        }
    }
    pthread_mutex_unlock(&globals->exec_complete_mutex);

    // debug_print(4,"retained me? at index %d, retain %d\n", shared_globals_index, globals->next_to_retain);
    if (globals->next_to_retain == shared_globals_index) {
        // Retain this particle
        debug_print(4,"Retaining trace ending in %d\n", getpid());
        for (int i=0; i<NUM_OBSERVES; i++) {
            globals->retained[i].retained_pid = locals->pid_trace[i];
        }
        pthread_mutex_lock(&(globals->retained_particle_set_mutex));
        globals->is_retained_particle_set = true;
        debug_print(3,"[broadcast retained_particle_set_cond] (particle %d)\n", globals->next_to_retain);
        pthread_cond_broadcast(&(globals->retained_particle_set_cond));
        pthread_mutex_unlock(&(globals->retained_particle_set_mutex));
    } else {
        // if not retained, exit.
        debug_print(4,"[%d -> not retained]\n", getpid());
    }
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
 * Special printf function "predict", for named doubles
 *
 */ 
void predict_value(const char *name, const double value) {
	if (IS_PRERUN) { return; }
    // generate "predict" queries, given name and value.
    utstring_printf(locals->predict,"%s,%f\n", name, value);
}

void weight_trace(const double ln_p, const bool synchronize) {

	// We do a single particle pre-run to count the number of synchronizing observes
	if (IS_PRERUN) {
		if (synchronize) NUM_OBSERVES++;
	} else {
		// It's not the prerun anymore. Now what?

        // If this isn't a synchronizing observe, we accumulate log probability 
        // and continue normal program execution.
   		if (!synchronize) {
 		    locals->log_weight += ln_p;
 		    return;
 		}

		// We want to branch and resample on every synchronizing observe
 		pthread_mutex_lock(&(globals->begin_observe_mutex));
 		int particles_to_count = NUM_PARTICLES - (globals->has_retained_particle ? 1 : 0);
 		int shared_globals_index = globals->begin_observe_counter;
 		locals->log_weight += ln_p;
 		globals->log_weights[shared_globals_index] = locals->log_weight;
 		globals->begin_observe_counter += 1;

        debug_print(4,"[OBSERVE %d, %d] #%d, %0.4f\n", locals->current_observe, getpid(), globals->begin_observe_counter, ln_p);

        // TODO check, fix

 		// Wait until processes are synchronized
        debug_print(3,"[observe #%d] #%d\n", locals->current_observe, globals->begin_observe_counter);
 		if (globals->begin_observe_counter >= particles_to_count) {
 			debug_print(4,"%d: observed %d of %d particles, moving on\n", getpid(), globals->begin_observe_counter, particles_to_count);

 			// Sample number of children
 			// Get update from retained particle, if there is one
 			if (globals->has_retained_particle) {
                // TODO check, fix
                debug_print(4,"YES THERE IS A RETAINED PARTICLE, it has log weight %f\n", globals->retained[locals->current_observe].retained_ln_p);
 				globals->log_weights[NUM_PARTICLES-1] = globals->retained[locals->current_observe].retained_ln_p; // retained_node->log_weight;
 			}
 
            // Reset observe counters to zero
            globals->begin_observe_counter = 0;
            globals->end_observe_counter = 0;
 
 			// sample offspring counts
	    multinomial_resample();
            //residual_resample();

 			// Signal retained node to create children
            if (globals->has_retained_particle) {
                debug_print(4,"Sending BRANCH to %d, at observe %d, hopefully\n", globals->retained[locals->current_observe].retained_pid, locals->current_observe);
                pthread_mutex_lock(&(globals->retained[locals->current_observe].branch_mutex));
                globals->retained[locals->current_observe].branch_flag = true;
                debug_print(3,"[broadcast retained[%d].branch_cond]\n", locals->current_observe);
                pthread_cond_broadcast(&(globals->retained[locals->current_observe].branch_cond));
                pthread_mutex_unlock(&(globals->retained[locals->current_observe].branch_mutex));
            }

            // Inform peer particles that synchronization for this observe is complete
            debug_print(3,"[broadcast begin_observe] observe = %d\n", locals->current_observe);
 			pthread_cond_broadcast(&globals->begin_observe_cond);
 		} else {
 			debug_print(4,"%d: observed %d of %d particles, waiting...\n", getpid(), globals->begin_observe_counter, particles_to_count);
 			// This *looks* strange, but the begin_observe_counter is incremented every 
 			// time this function is called, and then reset to zero before broadcast().
 			while (globals->begin_observe_counter != 0) {
                debug_print(3,"[wait begin_observe] observe barrier counter = %d (pid %d)\n", globals->begin_observe_counter, getpid());
     			pthread_cond_wait(&globals->begin_observe_cond, &globals->begin_observe_mutex);
     		}
 		}
 		pthread_mutex_unlock(&(globals->begin_observe_mutex));
 
        // Enter main control loop
        int n_offspring = globals->n_offspring[shared_globals_index];
        if (n_offspring > 0) {
            retain_branch_loop(n_offspring);
        } else {
            pthread_mutex_lock(&globals->end_observe_mutex);
            globals->end_observe_counter++;
            debug_print(3,"[end_observe] counter = %d (wait until %d) (%d had no children)\n", globals->end_observe_counter, NUM_PARTICLES, getpid());
            if(globals->end_observe_counter == NUM_PARTICLES) {
                pthread_cond_broadcast(&globals->end_observe_cond);
            }
            pthread_mutex_unlock(&globals->end_observe_mutex);
            destroy_particle();
        }
 		
 		// Wait until all particles have finished handling this observation
        pthread_mutex_lock(&globals->end_observe_mutex);
        while (globals->end_observe_counter < NUM_PARTICLES) {
            debug_print(3,"[wait end_observe] only seen %d of %d\n", globals->end_observe_counter, NUM_PARTICLES);
            pthread_cond_wait(&globals->end_observe_cond, &globals->end_observe_mutex);
        }
        pthread_mutex_unlock(&globals->end_observe_mutex);

        // Reset (local) log_weight for next observe
        locals->log_weight = 0;

	}
}


/**
 *
 * Initialize global state
 *
 */
void anglican_init_globals() {

    // Allocate shared memory
    globals = (shared_globals *)shared_memory_alloc(sizeof(shared_globals));
    globals->log_weights = (double *)shared_memory_alloc(NUM_PARTICLES*sizeof(double));
    globals->n_offspring = (int *)shared_memory_alloc(NUM_PARTICLES*sizeof(int));
    
    // Initialize process locks
    init_shared_mutex(&globals->exec_complete_mutex, &globals->exec_complete_cond);
    init_shared_mutex(&globals->begin_observe_mutex, &globals->begin_observe_cond);
    init_shared_mutex(&globals->end_observe_mutex, &globals->end_observe_cond);
    init_shared_mutex(&globals->retained_particle_set_mutex, &globals->retained_particle_set_cond);
    init_shared_mutex(&globals->retain_complete_mutex, &globals->retain_complete_cond);
    init_shared_mutex(&globals->stdout_mutex, NULL);
    
    // Initialize globals
    globals->begin_observe_counter = 0;
    globals->end_observe_counter = 0;
    globals->has_retained_particle = false;
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
    anglican_init_globals();

    // Create initial state (pre-fork)
    process_locals _locals;
    locals = &_locals;
    locals->live_offspring_count = 0;
    locals->current_observe = 0;
	utstring_new(locals->predict);

	// Do initial prerun (at the moment, all this does is count the number of observes)
    pid_t prerun_pid = fork();
    if (prerun_pid == 0) {
        f(argc, argv);
    	observe(0);
    	exit(NUM_OBSERVES);
    } else if (prerun_pid < 0) {
        perror("fork");
        exit(1);
    } else {
        int status = 0;
        pid_t terminated_pid = wait(&status);
        NUM_OBSERVES = WEXITSTATUS(status);
        assert(prerun_pid == terminated_pid);
    }
    
    
	
	debug_print(1, "Number of observes: %d\n", NUM_OBSERVES-1);

    // Get memory required for struct
    int mem_size = sizeof(shared_globals) + NUM_PARTICLES*(sizeof(double) + sizeof(int)) + (NUM_OBSERVES+1)*sizeof(retained_particle);
    debug_print(1, "Shared memory size: %d bytes\n", mem_size);

    // Allocate variables which depend on observe count
	locals->pid_trace = malloc(NUM_OBSERVES*sizeof(pid_t));
    globals->retained = (retained_particle *)shared_memory_alloc((NUM_OBSERVES+1)*sizeof(retained_particle));
    for (int index=0; index<NUM_OBSERVES; index++) {
       init_shared_mutex(&(globals->retained[index].branch_mutex), &(globals->retained[index].branch_cond) );
    }

    // Start timer
    struct timeval start_time;
    if (TIME_ITERATION) {
        gettimeofday(&start_time, NULL);
        debug_print(1, "Starting timer at %ld.%06d\n", start_time.tv_sec, (int)start_time.tv_usec);
    }

    // Run conditional SMC over and over a bunch of times
    IS_PRERUN = false;
    for (int iter=0; iter<NUM_ITERATIONS; iter++) {

#if DEBUG_LEVEL >= 3
       	debug_print(3,"\n----------\nPMCMC iteration %d\n----------\n", 1+iter);
#else
        debug_print(1, "PMCMC iteration %d of %d\n", 1+iter, NUM_ITERATIONS);
#endif

        globals->is_retained_particle_set = false;
        globals->exec_complete_counter = 0;
        globals->retain_complete_counter = 0;
        
        int particles_to_start = globals->has_retained_particle ? NUM_PARTICLES - 1 : NUM_PARTICLES;
        for (int i=0; i<particles_to_start; i++) {
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

                locals->pid_trace[0] = getpid();
                f(argc, argv);
                observe(0); // "dummy" observe to mark end of program.
                flush_output(&globals->stdout_mutex, locals->predict);
                
                set_retained_particle();

                destroy_particle();
            } else if (child_pid < 0) {
                // Error
                perror("fork");
                free(locals->pid_trace);
                utstring_free(locals->predict);
                exit(1);
            } else {
                locals->live_offspring_count++;
            }

            // This is the parent process
            assert(child_pid > 0);
        }
        
        // Chill out here until the retained particle has been set.
        pthread_mutex_lock(&(globals->retain_complete_mutex));
        while (globals->retain_complete_counter < NUM_OBSERVES) {
            debug_print(3,"[wait retain_complete] retained complete %d of %d\n", globals->retain_complete_counter, NUM_OBSERVES);
            pthread_cond_wait(&(globals->retain_complete_cond), &(globals->retain_complete_mutex));
        } 
        pthread_mutex_unlock(&(globals->retain_complete_mutex));
        debug_print(3,"retained particle set complete for iteration %d\n", iter);
        
#if DEBUG_LEVEL > 0
        if (iter == NUM_ITERATIONS - 1) {            
            fprintf(stderr, "NOTE: that was the last pmcmc iteration.\n");
        }
#endif

        // Collect terminated child processes
        debug_print(4,"Done launching particles -- waiting for %d of them to finish\n", locals->live_offspring_count-1);
        cleanup_children(locals->live_offspring_count-1, &locals->live_offspring_count);

        // Print out per-iteration timing info
        if (TIME_ITERATION) print_walltime(&globals->stdout_mutex, iter+1, &start_time);
    }

#if DEBUG_LEVEL >= 3
    fprintf(stderr, "\n------");
    fprintf(stderr, "All iterations complete. Releasing retained particle\n");
#endif

    // Release retained particle after last iteration
    globals->n_offspring[NUM_PARTICLES-1] = 0;
    for (int i=0; i<NUM_OBSERVES; i++) {
        globals->retained[i].retained_pid = -1;
        debug_print(4,"broadcast: releasing %d\n", i);
        
        pthread_mutex_lock(&(globals->retained[i].branch_mutex));
        globals->retained[i].branch_flag = true;
        debug_print(3,"[broadcast retained[%d].branch_cond] releasing retained particle\n", i);
        pthread_cond_broadcast(&(globals->retained[i].branch_cond));
        pthread_mutex_unlock(&(globals->retained[i].branch_mutex));
    }
    
    //printf("collecting last retained particle\n");
    wait(NULL);
    free(locals->pid_trace);
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
