#include "probabilistic.h"

int main(int argc, char **argv) {

    double var = 2;
    double mu = normal_rng(1, 5);

    printf("data[1],%f\n", normal_rng(mu, var)); 
    printf("data[2],%f\n", normal_rng(mu, var)); 
    
    printf("mu,%f\n", mu);

    return 0;
}

