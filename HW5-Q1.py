#!/usr/bin/env python2.7

# By: Nick Rocco and Ryan Loizzo
# Machine Learning Homework 5
# Problem 1

import numpy as np
import matplotlib.pyplot as plt
import math

weight_vector = [0.01, 0.01]

def target(x):
    val = -1 + float(x[0]) + float(x[1])
    if val > 0:
        return 1
    else:
        return -1

def direction(x):
    if x > 0:
        return 1
    else:
        return -1

def gradient_descent(data, learning_rate, ec, n_iters):
    global weight_vector
    iter_error = []
    sse = 0
    error = 0
    for i in range(n_iters):
        error = 0
        delta_w = [0,0]
        for entry in data:
            t = target(entry)
            o = direction(np.dot(entry, weight_vector))
            if ec:
                learning_rate = learning_rate/(i+1)
            delta_w += learning_rate * (t-o) * entry
            error += (t-o)*(t-o)
        weight_vector += delta_w
    
    print [n_iters, error] 
    return [n_iters, error]

if __name__ == '__main__':
    x_min_range = -1000
    x_max_range = 1000
    N = 5000
    m = 2
    data = np.random.uniform(x_min_range,x_max_range,size=[N,m])

    iters = [5, 10, 20, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
    
    iter_error = []

    for x in iters:
        l = gradient_descent(data, 0.001, False, x)
        iter_error.append(l)

    plt.plot([x[0] for x in iter_error], [y[1] for y in iter_error])
    plt.xlabel('Number of iterations')
    plt.ylabel('Sum of Squared Error')
    plt.show()
