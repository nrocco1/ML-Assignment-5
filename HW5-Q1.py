#!/usr/bin/env python2.7

# By: Nick Rocco and Ryan Loizzo
# Machine Learning Homework 5
# Problem 1

import numpy as np
import matplotlib.pyplot as plt
import math

def target(x):
    val = -1 + float(x[1]) + float(x[2])
    if val > 0:
        return 1
    else:
        return -1

def direction(x):
    if x > 0:
        return 1
    else:
        return -1

def gradient_descent(data, learning_rate):
    count = 0
    iter_error = []
    sse = 0
    error = 0
    weight_vector = [0.01, 0.01, 0.01]
    delta_w = 0
    for entry in data:
        t = target(entry)
        o = direction(np.dot(entry, weight_vector))
        count += 1
        delta_w += learning_rate * (t-o) * entry
        weight_vector += delta_w
        error += (t-o)*(t-o)
        if count == 5:
            iter_error.append([count, error])
        elif count == 10:
            iter_error.append([count, error])
        elif count == 20:
            iter_error.append([count, error])
        elif count == 50:
            iter_error.append([count, error])
        elif count == 100:
            iter_error.append([count, error])
        elif count == 500:
            iter_error.append([count, error])
        elif count == 1000:
            iter_error.append([count, error])
        elif count == 2000:
            iter_error.append([count, error])
        elif count == 3000:
            iter_error.append([count, error])
        elif count == 4000:
            iter_error.append([count, error])
        elif count == 5000:
            iter_error.append([count, error])
    
    return iter_error

if __name__ == '__main__':
    x_min_range = -1000
    x_max_range = 1000
    N = 5000
    m = 2
    data = np.random.uniform(x_min_range,x_max_range,size=[N,m])
    data = np.concatenate((np.ones([N,1]),data),axis=1)

    iter_error = gradient_descent(data, 0.05)

    plt.plot([x[0] for x in iter_error], [y[1] for y in iter_error])
    plt.xlabel('Number of iterations')
    plt.ylabel('Sum of Squared Error')
    plt.show()
