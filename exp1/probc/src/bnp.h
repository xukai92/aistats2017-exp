#ifndef __BNP__

// Polya urn (CRP)

typedef struct {
    double concentration;
    int len_buckets;
    int max_buckets;
    int sum_counts;
    int *counts;
} polya_urn_state;

/**
 * Create a polya urn with specified concentration
 *
 */
void polya_urn_new(polya_urn_state *state, double concentration);

/**
 * Clear polya urn state, free internal memory usage
 *
 */
void polya_urn_free(polya_urn_state *state);

/**
 * Draw a number from an existing urn
 *
 */
int polya_urn_draw(polya_urn_state *state);



// Stick-breaking process

typedef struct {
    double concentration;
    int len_buckets;
    int max_buckets;
    double beta_prod;
    double beta_sum;
    double *beta;
} stick_dist;

void stick_new(stick_dist *state, double concentration);
void stick_free(stick_dist *state);
int stick_rng(stick_dist *state);

#define __BNP__
#endif