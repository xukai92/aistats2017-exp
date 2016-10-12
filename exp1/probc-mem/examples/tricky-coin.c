#include "probabilistic.h"

int main(int argc, char** argv) {

    /* 
     * "tricky coin" example from Venture documentation
     * http://probcomp.csail.mit.edu/venture/console-examples.html
     *
     */

    // p(is_tricky) = 0.1
    int is_tricky = flip_rng(0.1);

    // theta | is_tricky ~ beta(1,1)
    // theta | !is_tricky = 0.5
    double theta = is_tricky ? beta_rng(1, 1) : 0.5;

    // observe 5 coin flips, all coming up heads
    observe(flip_lnp(1, theta));
    observe(flip_lnp(1, theta));
    observe(flip_lnp(1, theta));
    observe(flip_lnp(1, theta));
    observe(flip_lnp(1, theta));

    // is the coin tricky?
    predict("is_tricky,%d\n", is_tricky);

    // what percent of the time does the coin come up heads?
    predict("theta,%0.4f\n", theta);
 
    return 0;
}
