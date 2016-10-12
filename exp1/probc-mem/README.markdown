Probabilistic C
==================

Reference implementation of [A Compilation Target for Probabilistic Programming Languages](http://arxiv.org/abs/1403.0504). 

With probabilistic C, you can run particle Markov chain Monte Carlo inference over arbitrary C code.

Installation
--------------

This has been tested on OS X (10.8, 10.9) and on Ubuntu Linux (12.04, 13.04) and should work on any Unix or Unix-like operating system that supports the `fork` system call.

To get started, clone the repository and run `make`. 
This will build the library, linking against the particle Gibbs engine, and will compile the example programs in `examples/`.

To run a simple example, try:

    ./bin/gaussian-unknown-mean -p 100 -i 100 | ./compute_moments.sh

This will run the Gaussian with unknown mean example (in `examples/gaussian-unknown-mean.c`).
Look at the code to see the model definition.
The helper script `compute_moments.sh` prints out the first and second moments of each `predict` value we are sampling.
There is an additional helper script, `compute_counts.sh`, which may be more appropriate for discrete data.

### Configuration on Linux

The degree to which the PMCMC sampler mixes depends a lot on how many particles we can run simultaneously in each sweep.
This, in turn, depends on the number of simultaneous processes our operating system can handle.
The following configuration options (tested on Ubuntu 12.04, 12.10) greatly increase the amount of simultaneous processes permitted by the kernel:

1. Add the following to /etc/sysctl.conf:

    kernel.pid_max = 4194303

2. Add the following to /etc/security/limits.conf:

'''
    *       soft    nproc   unlimited
    *       hard    nproc   unlimited
'''

If you try to run too many simultaneous particles, you will know it by seeing a stream of messages along the lines of

    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable
    fork: Resource temporarily unavailable

The total number of processes required to run a given number of particles varies by model
and inference engine. Particle Gibbs has the highest process count requirements, which 
in the worse case should be bounded by

1 + (1 + # of blocking observes) x (# of particles per sweep)

PIMH and SMC only require order # of particles processes.
The particle cascade algorithm runs in a fixed process count budget.


### Configuration on Mac OS X

Unfortunately limits on non-server versions of OS X prevent us from setting `nproc unlimited`.
We can still bump up the process limit to 2500, though.

(documentation todo)


### Configuration on Microsoft Windows

In theory, this should work under cygwin, but this has never been tested.
Performance will probably be quite poor --- the process model in Windows is somewhat different
from in POSIX systems, and each call to `fork` will probably end up deep copying the entire
program memory state.

If anyone tries to run this on Windows, please report how and if it works.


### Alternate inference engines

Alternate inference engines just need to implement `probabilistic.h`.
At the moment, the default is particle Gibbs, in `src/pg.c`.
To compile the examples (and the library) with an alternate inference engine,
the particle cascade (asynchronous anytime SMC, as introduced in [http://arxiv.org/abs/1407.2864](http://arxiv.org/abs/1407.2864)),
try

    make ENGINE=cascade

For particle independent Metropolis-Hastings, try

    make ENGINE=pimh

Alternately, you can just run a single SMC sweep with a fixed number of particles with

    make ENGINE=smc

More inference backends are on the way.

Note that the output from the particle cascade differs in format from the output from
the particle MCMC algorithms; the particle cascade prints out *weighted* values.
That is, in the example programs each line of output from the particle Gibbs engine looks like

    mu,6.027172

in the format `variable,value`. 
In contrast, output from the particle cascade for the same example program 

    mu,7.488961,-7.721001,8569
    
is formatted `variable,value,weight,id`.
The `id` is a unique identifier for a single execution of the program, and can be used to
link together multiple variables.
The `weight` is an unnormalized log importance weight.
To compute the actual "weight" of the sample, we must exponentiate these and normalize by
their sum.
A simple python script `coallate_weights.py` is included, demonstrating how to normalize
the weights for marginal distributions of the different sampled variables.
For example, running

    ./bin/gaussian-unknown-mean -p 10000 | python coallate_weights.py 
    
will output lines such as 

    mu,6.198685,0.001553
    
where the last number is a (normalized) importance weight for this particular sampled value
of `mu`.


Writing and editing programs
------------------------------

### Probabilistic C language definition

It's plain C, plus 2 (or really, 4) keywords defined in `src/probabilistic.h`.

* `void observe(const double ln_p)` weights an execution trace by a log-probability, introducing a resampling point where program execution will fork
* `void predict(const char *format, ...)`
is a synchronized print function (with `printf` semantics).
Anything which is printed in this manner will print valid samples from the posterior distribution of the model defined by the program and weighted by the execution trace.

There are two alternative versions of these functions which may be useful in some contexts:

* `void weight_trace(const double ln_p, const bool synchronize)` weights an execution trace 
by a log-probability `ln_p`, but with an added `synchronize` flag that determines whether 
or not we will block and resample on this particular data point. 
Calling `observe` is equivalent to calling `weight_trace` with `synchronize = true`.
* `void predict_value(const char *name, const double value)` is a shorthand for predicting 
real-valued quantities; equivalent to `predict('%s,%f\n', name, value)`.

### Random number generators and log-density functions

There are a number of useful random number generators and log probability functions included; see `src/erp.h`.
Many of these are wrappers for random number generators normally bundled with NumPy (see `ext/`).
There is no requirement to use these: randomness, and execution trace weights, may be generated in any manner desired,
with some caveats regarding updating seeds for pseudorandom number generators after a process forks.

An example implementation of some simple nonparametric models is also included, in `src/bnp.h` and `src/bnp.c`; see `examples/crp.c` to see it in action.


### Usage as a compilation target

The general-purpose function memoization construct in `src/memoize.h` may come in handy,
particularly when compiling from other languages, particularly from scheme-like languages such as Church, Anglican, or Venture.
See `examples/crp.c` for sample usage.





License
---------

Copyright © 2014, 2015 Wood group

This file is part of Probabilistic C, a probabilistic programming system.

Probabilistic C is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Probabilistic C is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the [GNU General Public
License](gpl-3.0.txt) along with Probabilistic C.  If not, see
[http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
