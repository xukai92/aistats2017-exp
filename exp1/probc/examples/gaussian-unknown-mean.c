#include "probabilistic.h"

int main(int argc, char **argv) {

    double var = 2;
    double mu = normal_rng(1, 5);

    observe(normal_lnp(9, mu, var)); 
    observe(normal_lnp(8, mu, var)); 
    
    predict("mu,%f\n", mu);

    return 0;
}
