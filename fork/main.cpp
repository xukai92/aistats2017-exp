#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>

#include <ctime>

#include <iostream>
#include <fstream>

int main() {
    using namespace std;

    const int NUM = 10;     // number of processes to fork
    const int DEPTH = 50;  // depth of the Fibonacci sequence
    int seq[DEPTH];         // an array to store the Fib seq
                            // this may need to be changed to a linked list based stack
    clock_t start, end;     // variable to record the start and end time
    pid_t pid;              // process id

    ofstream file ("/Users/kai/Turing/exps/aistats2017/fork/process.csv");

    for (int n = 0; n < DEPTH; n++) {

        // Compute the Fibonacci sequence
        if (n == 0 or n == 1) seq[n] = 1;
        else                  seq[n] = seq[n - 1] + seq[n - 2];

        // Fork a batch of processes
        for (int ith = 0; ith < NUM; ith++) {
            start = clock();    // record the start time
            pid = fork();       // fork the process
            end = clock();      // record the end time

            if (pid) {
                // Kill the forked processes
                if (pid != -1) kill(pid, SIGKILL);
                // Write the elapsed time to the CSV file
                file << double(end - start) / CLOCKS_PER_SEC;
                if (ith < NUM - 1) file << ';';
            } else {
                pause();
            }
        }
        file << endl;           // line break
    }


    file.close();               // close the CSV file
    return 0;
}

