#include "probabilistic.h"
#define K 3
#define N 11

/* Markov transition matrix */
static double T[K][K] = { { 0.1,  0.5,  0.4 }, 
                          { 0.2,  0.2,  0.6 }, 
                          { 0.15, 0.15, 0.7 } };





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
            printf("data[%d],%f\n", n, normal_rng(state_mean[states[n]], 1));
        }
        printf("state[%d],%d\n", n, states[n]);
    }
    
    return 0;
}
