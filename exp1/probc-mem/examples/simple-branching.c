#include "probabilistic.h"

int fib(int n) {
    return (n < 2) ? n : fib(n-1) + fib(n-2);
}

int main(int argc, char **argv) {
    int r = poisson_rng(4);
    int l = (4 < r) ? 6 : fib(3*r) + poisson_rng(4);
    observe(poisson_lnp(6, l));
    predict("r,%3d\n", r);
    return 0;
}
