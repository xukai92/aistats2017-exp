#include "probabilistic.h"
#define K 3
#define N 17

/* Markov transition matrix */
static double T[K][K] = {{ 0.1,  0.5,  0.4 }, 
                         { 0.2,  0.2,  0.6 }, 
                         { 0.15, 0.15, 0.7 }};


/* Observed data */
static double data[N] = { NAN, .9, .8, .7, 0, -.025,
                               -5, -2, -.1, 0, 0.13, 0.45, 
                                6, 0.2, 0.3, -1, -1 };

/* Prior distribution on initial state */
static double initial_state[K] = { 1.0/3, 1.0/3, 1.0/3 };

/* Per-state mean of Gaussian emission distribution */
static double state_mean[K] = { -1, 1, 0 };

/* Generative program for a HMM */
int main(int argc, char **argv) {
    
    int states[N];
    for (int n=0; n<N; n++) {
        states[n] = (n==0) ? discrete_rng(initial_state, K) 
                           : discrete_rng(T[states[n-1]], K);
        if (n > 0) {
            observe(normal_lnp(data[n], 
            		       state_mean[states[n]], 1));
        }
        predict("state[%d],%d\n", n, states[n]);
    }
    
    return 0;
}
