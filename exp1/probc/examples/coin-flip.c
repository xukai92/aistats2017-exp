#include "probabilistic.h"

int main(int argc, char** argv) {

    /* true posterior p ~ beta(3,2)  => E[p] = 0.6 */
    double p = beta_rng(1, 1);
    observe(flip_lnp(1, p));
    observe(flip_lnp(1, p));
    observe(flip_lnp(0, p));
    predict("p,%0.4f\n", p);
 
    return 0;
}
