#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "bnp.h"
#include "erp.h"


// Polya urn

void polya_urn_new(polya_urn_state *state, double concentration) {
    int s = 2; // initial size
    *state = (polya_urn_state) { concentration, 0, s, 0, malloc(s*sizeof(int)) };
}

void polya_urn_free(polya_urn_state *state) {
    free(state->counts);
}

int polya_urn_draw(polya_urn_state *state) {
    // expand internal state if necessary
    if (state->len_buckets == state->max_buckets) {
        state->max_buckets *= 2; // growth factor
        state->counts = realloc(state->counts, state->max_buckets*sizeof(int));
    }
    
    // draw from urn
    if (state->len_buckets == 0) {
        // first draw
        state->counts[0] = 1;
        state->len_buckets = 1;
        state->sum_counts++;
        return 0;
    } else {
        // subsequent draws
        double *sampling_dist = malloc((1+state->len_buckets)*sizeof(double));
        for (int i=0; i<state->len_buckets; i++) {
            sampling_dist[i] = (double)state->counts[i] / (state->concentration + state->sum_counts);
        }
        sampling_dist[state->len_buckets] = state->concentration / (state->concentration + state->sum_counts);
        
        // draw
        int bucket = discrete_rng(sampling_dist, state->len_buckets+1);
        free(sampling_dist);

        // update counts
        if (bucket < state->len_buckets) {
            state->counts[bucket]++;
        } else {
            state->counts[bucket] = 1;
            state->len_buckets++;
        }
        state->sum_counts++;
        return bucket;
    }
}


/// Stick breaking

void stick_new(stick_dist *state, double concentration) {
    int s = 2; // initial size
    *state = (stick_dist) { concentration, -1, s, 1.0, 0.0, malloc(s*sizeof(double)) };
}

void stick_free(stick_dist *state) {
    free(state->beta);
}

int stick_rng(stick_dist *state) {
    // draw number between 0 and 1
    double u = uniform_rng(0, 1);
    // find entry
    int entry = 0;
    while (true) {
        if (entry == state->max_buckets) {
            // expand internal state if necessary
            state->max_buckets *= 2; // growth factor
            state->beta = realloc(state->beta, state->max_buckets*sizeof(double));
        }
        if (entry > state->len_buckets) {
            // compute new entries as needed
            // TODO consider do this all in log space
            double beta_prime = beta_rng(1, state->concentration);
            double beta_next = beta_prime * state->beta_prod;
            state->beta[entry] = beta_next + state->beta_sum;
//             fprintf(stderr, "new value for entry[%d]: %f\n", entry, state->beta[entry]);
            state->beta_prod *= (1-beta_prime);
            state->beta_sum += beta_next;
            state->len_buckets++;
        }
        if (state->beta[entry] > u) {
//             fprintf(stderr, "returning entry[%d] = %f; log_u = %f\n", entry, state->beta[entry], u);
            return entry;
        }
        entry++;
        if (entry > 50) { exit(1); }
    }
}
