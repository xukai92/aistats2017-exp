#include "probabilistic.h"
#define N 10

// Observed data
static double data[N] = { 1.0,  1.1,   1.2,
                         -1.0, -1.5,  -2.0, 
                        0.001, 0.01, 0.005, 0.0 };

// Struct holding mean and variance parameters for each cluster
typedef struct theta {
    double mu;
    double var;
} theta;

// Draws a sample of theta from a normal-gamma prior
theta draw_theta() {
    double variance = 1.0 / gamma_rng(1, 1);
    return (theta) { normal_rng(0, variance), variance };
}

// Get the class id for a given observation index
static polya_urn_state urn;
void get_class(int *index, int *class_id) {
    *class_id = polya_urn_draw(&urn);
}

int main(int argc, char **argv) {
    double alpha = 1.0;
    polya_urn_new(&urn, alpha);

    mem_func mem_get_class; 
    memoize(&mem_get_class, get_class, sizeof(int), sizeof(int));

    theta params[N];
    bool known_params[N] = { false };

    int class;
    for (int n=0; n<N; n++) {
        mem_invoke(&mem_get_class, &n, &class);
        if (!known_params[class]) {
            params[class] = draw_theta();
            known_params[class] = true;
        }
        observe(normal_lnp(data[n], params[class].mu, 
                                    params[class].var));
    }

    // Predict number of classes
    predict("num_classes,%2d\n", urn.len_buckets);

    // Release memory; exit
    polya_urn_free(&urn);
    return 0;
}
