#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#include "distributions.h"
#include "erp.h"

static rk_state state;

void erp_rng_init() {
    rk_seed(time(NULL), &state);
}

unsigned long int gen_new_rng_seed() {
    return rk_random(&state);
}

void set_rng_seed(unsigned long int seed) {
    rk_seed(seed, &state);
}


/*
 * Begin functions for probability distributions
 */

unsigned int flip_rng(double p) {
    return (rk_uniform(&state, 0, 1) < p) ? 1 : 0;
}

double flip_lnp(int x, double p) {
    // TODO check that x in {0,1}
    return log((x==1) ? p : 1 - p);
}

long poisson_rng(double rate) {
    return rk_poisson(&state, rate);
}

double poisson_lnp(long x, double rate) {
    // This is slightly approximate, I suppose
    return x*log(rate) - rate - lgamma((double)(x+1));
}

double gamma_rng(double shape, double rate) {
    return rk_standard_gamma(&state, shape) / rate;
}

double gamma_lnp(double x, double shape, double rate) {
    return shape*log(rate) - lgamma(shape) + (shape - 1)*log(x) - rate*x;
}

double beta_rng(double a, double b) {
    return rk_beta(&state, a, b);
}

double beta_lnp(double x, double a, double b) {
    const double Z = lgamma(a) + lgamma(b) - lgamma(a+b);
    return (a-1)*log(x) + (b-1)*log(1-x) - Z;
}

double normal_rng(double mean, double variance) {
    const double sd = sqrt(variance);
    return rk_normal(&state, mean, sd);
}

double normal_lnp(double x, double mean, double variance) {
    const double xmms = pow(x-mean, 2);
    const double pi = M_PI;
    const double Z = 0.5*log(2*pi*variance);
    return -0.5*xmms/variance - Z;
}

int uniform_discrete_rng(int num_elements) {
    return rk_interval(num_elements-1, &state);
}

double uniform_discrete_lnp(int x, int num_elements) {
    if (x < 0 || x >= num_elements) {
        return -INFINITY;
    } else {
        return -log(num_elements);
    }
}

double uniform_rng(double lower, double upper) {
    return rk_uniform(&state, lower, upper);
}

double uniform_lnp(double x, double lower, double upper) {
    return -log(upper - lower);
}

long sample_long_rng() {
    return rk_long(&state);
}

unsigned int discrete_rng(double *p, int K) {
    const double u = rk_uniform(&state, 0, 1);
    double sum = 0;
    for (int k=0; k<K; k++) {
        sum += p[k];
    }
    //printf("sum = %f\n", sum);
    //assert(sum == 1.0);

    double cumsum = 0;
    for (int k=0; k<K; k++) {
        cumsum += p[k]/sum;
        if (u < cumsum) {
            return k;
        }
    }
    fprintf(stderr, "[ERROR] cumsum = %f, u = %f, K = %d\n", cumsum, u, K);
    for (int k=0; k<K; k++) {
        fprintf(stderr, "%f ", p[k]);
    }
    fprintf(stderr, "\n");
    // assert(abs(cumsum - 1.0) < 0.0001);
    return uniform_discrete_rng(K);
}

double discrete_lnp(int x, double *p, int K) {
    if (x < 0 || x >= K) {
        return 0;
    } else {
        return log(p[x]);
    }
}

unsigned int discrete_log_rng(double *log_p, int K) {
    // log_p might be unnormalized
//     double A = log_p[0];
//     for (int k=1; k<K; k++) {
//         if (A < log_p[k]) A = log_p[k];
//     }
//     double log_normalizer = A;
//     for (int k=0; k<K; k++) {
//         
//     }
    const double u = rk_uniform(&state, 0, 1);
    double cumsum = 0;
    for (int k=0; k<K; k++) {
        cumsum += exp(log_p[k]);
        if (u < cumsum) {
            return k;
        }
    }
    fprintf(stderr, "[ERROR] cumsum = %f, u = %f, K = %d\n", cumsum, u, K);
    for (int k=0; k<K; k++) {
        fprintf(stderr, "%f ", log_p[k]);
    }
    fprintf(stderr, "\n");
    return uniform_discrete_rng(K);
}


// Multivariate distributions

void dirichlet_rng(double *x, double *alpha, int K) {
    double sum = 0;
    for (int k=0; k<K; k++) {
        x[k] = gamma_rng(alpha[k], 1);
        sum += x[k];
    }
    for (int k=0; k<K; k++) x[k] /= sum;
}

double dirichlet_lnp(double *x, double *alpha, int K) {
    double ln_p = 0; 
    double sum_alpha = 0;
    for (int k=0; k<K; k++) {
        sum_alpha += alpha[k];
        ln_p += (alpha[k]-1)*log(x[k]) - lgamma(alpha[k]);
    }
    return ln_p + lgamma(sum_alpha);
}

void dirichlet_sym_rng(double *x, double alpha, int K) {
    double sum = 0;
    for (int k=0; k<K; k++) {
        x[k] = gamma_rng(alpha, 1);
        sum += x[k];
    }
    for (int k=0; k<K; k++) x[k] /= sum;
}

double dirichlet_sym_lnp(double *x, double alpha, int K) {
    double ln_p = lgamma(K*alpha) - K*lgamma(alpha);
    for (int k=0; k<K; k++) {
        ln_p += (alpha-1)*log(x[k]);
    }
    return ln_p;
}

void dirichlet_sym_log_rng(double *log_x, double alpha, int K) {
   double sum = 0;
    for (int k=0; k<K; k++) {
        double entry = 0;
//         while (entry == 0)
        entry = rk_standard_gamma(&state, alpha);
        log_x[k] = log(entry);
        if (isinf(log_x[k])) {
            fprintf(stderr, "LOG ENTRY IS -INF: %d, value = %0.8f\n", k, entry);
        }
        assert(!isinf(log_x[k]));
        sum += entry;
    }
    double log_sum = log(sum);
    assert(!isinf(log_sum));
    for (int k=0; k<K; k++) log_x[k] -= log_sum;
}
