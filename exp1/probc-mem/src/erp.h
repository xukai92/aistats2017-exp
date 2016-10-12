#ifndef __ERP__
#define __ERP__


/* initialize ERP random number generator */
void erp_rng_init();

/* reseed ERP generator */
unsigned long int gen_new_rng_seed();
void set_rng_seed(unsigned long int seed);


/* flip */
unsigned int flip_rng(double p);
double flip_lnp(int x, double p);

/* discrete */
unsigned int discrete_rng(double *p, int K);
double discrete_lnp(int x, double *p, int K);

/* discrete variant, takes log-probability vector */
unsigned int discrete_log_rng(double *log_p, int K);

/* uniform continuous on half-open interval [lower, upper) */
double uniform_rng(double lower, double upper);
double uniform_lnp(double x, double lower, double upper);

/* uniform-discrete */
int uniform_discrete_rng(int num_elements);
double uniform_discrete_lnp(int x, int num_elements);

/* poisson */
long poisson_rng(double rate);
double poisson_lnp(long x, double rate);

/* gamma */
double gamma_rng(double shape, double rate);
double gamma_lnp(double x, double shape, double rate);

/* beta */
double beta_rng(double a, double b);
double beta_lnp(double x, double a, double b);

/* normal */
double normal_rng(double mean, double variance);
double normal_lnp(double x, double mean, double variance);

/* samples a random long-typed value from 0 and LONG_MAX inclusive */
long sample_long_rng();

/* begin multivariate distributions */

/* dirichlet */
void dirichlet_rng(double *x, double *alpha, int K);
double dirichlet_lnp(double *x, double *alpha, int K);

/* symmetric dirichlet */
void dirichlet_sym_rng(double *x, double alpha, int K);
double dirichlet_sym_lnp(double *x, double alpha, int K);

/* symmetric dirichlet variant, returns log-entries */
void dirichlet_sym_log_rng(double *log_x, double alpha, int K);

#endif
