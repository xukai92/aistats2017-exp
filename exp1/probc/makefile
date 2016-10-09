ENGINE=pg
VERBOSITY=0
INTERVAL=1
OPTI= -O3 -finline-functions -fomit-frame-pointer \
-fno-strict-aliasing --param max-inline-insns-single=1800
CC=gcc -std=gnu99 -Wall -O3 -ffast-math -fomit-frame-pointer -finline-functions
CPP=g++ -std=c++11 -Wall -g
ODIR=bin/
UNAME:=$(shell uname)
OBJ=ext/mtrand/randomkit.o ext/mtrand/distributions.o src/engine-shared.o src/erp.o src/engine.o src/memoize.o src/bnp.o
LIBPROB=$(ODIR)libprob.a
HEADERS=-Isrc/ -Iext/mtrand/ -Iext/uthash/src/
ifeq ($(UNAME), Darwin)
LIBS=-lpthread -lm
else
LIBS=-lpthread -lm -lrt
endif

all: engine examples
	
examples: gaussian-unknown-mean coin-flip tricky-coin hmm big-hmm linear-gaussian crp simple-branching priors

$(ODIR):
	test -d $(ODIR) || mkdir $(ODIR)

mtrand: ext/mtrand/*.c | $(ODIR)
	cd ext/mtrand && $(CC) -c randomkit.c distributions.c 

engine: src/*.c mtrand | $(ODIR)
	$(CC) -c src/erp.c -o src/erp.o $(HEADERS)
	$(CC) -c src/bnp.c -o src/bnp.o $(HEADERS)
	$(CC) -c src/memoize.c -o src/memoize.o $(HEADERS)
	$(CC) -c src/engine-shared.c -o src/engine-shared.o $(HEADERS)
	$(CC) -c src/$(ENGINE).c -o src/engine.o $(HEADERS) -DDEBUG_LEVEL=$(VERBOSITY)
	ar rcs $(LIBPROB) $(OBJ)

crp: examples/crp.c engine | $(ODIR)
	$(CC) -o $(ODIR)crp examples/crp.c $(LIBPROB) $(LIBS) $(HEADERS)

simple-branching: examples/simple-branching.c engine | $(ODIR)
	$(CC) -o $(ODIR)simple-branching examples/simple-branching.c $(LIBPROB) $(LIBS) $(HEADERS)

hmm: examples/hmm.c engine | $(ODIR)
	$(CC) -o $(ODIR)hmm examples/hmm.c -DINTERVAL=$(INTERVAL) $(LIBPROB) $(LIBS) $(HEADERS)

big-hmm: engine | $(ODIR)
	$(CC) -o $(ODIR)big-hmm examples/big-hmm.c -DINTERVAL=$(INTERVAL) $(LIBPROB) $(LIBS) $(HEADERS)

linear-gaussian: engine | $(ODIR)
	$(CC) -o $(ODIR)linear-gaussian examples/linear-gaussian-1d.c -DINTERVAL=$(INTERVAL) $(LIBPROB) $(LIBS) $(HEADERS)

gaussian-unknown-mean: examples/gaussian-unknown-mean.c engine | $(ODIR)
	$(CC) -o $(ODIR)gaussian-unknown-mean examples/gaussian-unknown-mean.c $(LIBPROB) $(LIBS) $(HEADERS)

coin-flip: examples/coin-flip.c engine | $(ODIR)
	$(CC) -o $(ODIR)coin-flip examples/coin-flip.c $(LIBPROB) $(LIBS) $(HEADERS)

tricky-coin: examples/tricky-coin.c engine | $(ODIR)
	$(CC) -o $(ODIR)tricky-coin examples/tricky-coin.c $(LIBPROB) $(LIBS) $(HEADERS)

priors: examples/gaussian-prior.c examples/hmm-prior.c engine | $(ODIR)
	$(CC) -o $(ODIR)gaussian-prior examples/gaussian-prior.c $(LIBPROB) $(LIBS) $(HEADERS)
	$(CC) -o $(ODIR)hmm-prior examples/hmm-prior.c $(LIBPROB) $(LIBS) $(HEADERS)

clean:
	rm -f ext/mtrand/*.o
	rm -f src/*.o
	rm -f $(LIBPROB)
