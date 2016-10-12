#!/usr/bin/env python

import csv
import fileinput
import numpy as np

from collections import defaultdict


def log_sum_exp(log_weights):
    '''
    uses stupid "log sum exp" trick to attempt to avoid underflow
    '''
    A = log_weights.max();
    return A + np.log(np.sum(np.exp(log_weights - A)));


if __name__ == '__main__':

    particles = defaultdict(int)
    log_weights = dict()

    for row in csv.reader(fileinput.input()):
        particles[(row[0], row[1], float(row[2]))] += 1

    for key in particles.keys():
        ident, value, weight = key
        count = particles[key]
        weight_key = (ident, value)
        if weight_key not in log_weights.keys():
            log_weights[weight_key] = weight + np.log(count)
        else:
            new_weight = weight + np.log(count)
            log_weights[weight_key] = log_sum_exp(np.array([log_weights[weight_key], new_weight]))
    
    const = log_sum_exp(np.array(log_weights.values()))
    ident_weights = defaultdict(list)
    for key in log_weights.keys():
        ident, value = key
        log_weights[key] = log_weights[key] - const
        ident_weights[ident].append(log_weights[key])

    for key in log_weights.keys():
        print '%s,%s,%f' % (key[0], key[1], np.exp(log_weights[key] - log_sum_exp(np.array(ident_weights[ident]))))
