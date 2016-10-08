#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

#include <ctime>

#include <iostream>
#include <fstream>

int main() {
    using namespace std;

    const int NUM = 100;    // number of processes to fork
    const int DEPTH = 100;  // depth of the Fibonacci sequence

    int seq[DEPTH];         // an array to store the Fib seq
                            // this may need to be changed to a linked list based stack
    clock_t start;          // variable to record the start time

    ofstream file ("/Users/kai/Turing/exps/aistats2017/fork/process.csv");

    for (int n = 0; n < DEPTH; n++) {

        // Compute the Fibonacci sequence
        if (n == 0 or n == 1) {
            seq[n] = 1;
        } else {
            seq[n] = seq[n - 1] + seq[n - 2];
        }

        double times[NUM];      // container to store elapsed times

        // Fork a batch of processes
        pid_t pid;
        for (int ith = 0; ith < NUM; ith++) {
            start = clock();
            pid = fork();
            if (pid) {
                file << double(clock() - start) / CLOCKS_PER_SEC;
                if (ith < NUM - 1) file << ';';
                continue;
            } else {
                break;
            }
        }
        file << endl;

        // Terminate the child processes immediately after creating
        // Since we only want to measure the time creating them
        if (pid) {
            continue;
        } else {
            break;
        }

    }

    file.close();

    return 0;
}

