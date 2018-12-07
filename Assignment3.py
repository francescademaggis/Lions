#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:42:00 2018

@author: demaggis
"""

import pandas as pd
import numpy as np

# importing the data

data = pd.read_csv('bigcity.csv',sep=',')

# dropping missing values 
data.dropna(how = "all", inplace=True)
data.head()

# -------------------------------------------------------------------------------------------------------

## EXERCISE 1

# In order to produce a good estimate of theta (and make it converge to the real value of the population's
# paramenter), we can carry out a simple Boostrapping procedure: for 10'000 times we randomly extract 29 obs.
# from the given empirical sample and for each of the variable (u and x) that we need to compute the estimates:

data_u = data["u"]
data_u.head()
data_x = data["x"]
data_x.head()

bs_sample_u = np.random.choice(data_u, size=10000)
np.mean(bs_sample_u)


rep = 10000
BT_sample_u = np.zeros(rep)
BT_sample_x = np.zeros(rep)
estimates = np.zeros(rep)

for i in range(rep):
    BT_sample_x[i] = np.random.choice(data_x, len(data)).mean()
    BT_sample_u[i] = np.random.choice(data_u, len(data)).mean()
    estimates[i] = BT_sample_x[i]/BT_sample_u[i]
    
bias = estimates.mean() - data_x.mean() / data_u.mean()
bias

se = estimates.std()
se

# 0.9 confidence interval
CI = np.percentile(estimates, [5,95])
CI

# -------------------------------------------------------------------------------------------------------

## EXERCISE 2

def Weibull_inverse(F, alpha=1, mu=1):
    y = mu * (- np.log(1 - F)) ** (1 / alpha)
    return y

rep2 = 10000
pseudo_rnd_dist = np.zeros(rep2)

for i in range(rep2):
    pseudo_rnd_dist[i] = Weibull_inverse(np.random.uniform())

# Plotting our sample
import seaborn as sns
sns.distplot(pseudo_rnd_dist, hist=False)

# Plotting the theoretical Weibull dist
sns.distplot(np.random.weibull(1, rep2), hist=False)

# by running line 73 and 76 together, we see a perfect matching of the two curves.
# This is the evidence that the random values generated using the Weibull inverse cumulative dist function 
# have themselves a Weibull distribution.

# -------------------------------------------------------------------------------------------------------

## EXERCISE 3

import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab


def gaussian(x, mean, dev = 0.1):
    # standard deviation, square root of variance
    return 1 / math.sqrt(2 * math.pi) / dev * math.exp( -(x - mean) ** 2 / 2 / (dev ** 2) )

# Generating data
N = 1000
a = 0.3

sample1 = np.random.normal(0, 0.2, size = int(N * a))
sample2 = np.random.normal(3, 0.1, size = int(N * (1 - a)))

sample = np.concatenate([sample1,sample2])


# Learning parameters
max_iter = 50

# Initial guess of parameters and initializations
params = np.array([-2, 1, 0.5, 1, 2])

# master loop
counter = 0
converged = False
tol = 0.1

plabel1 = np.zeros(sample.shape)
plabel2 = np.zeros(sample.shape)


while not converged and counter < 100:
    counter += 1
    
    mu1, mu2, pi_1, sd1, sd2 = params

    # Expectation
    for i in range(len(sample)):
        cdf1 = gaussian(sample[i], mu1, sd1)
        cdf2 = gaussian(sample[i], mu2, sd2)

        pi_2 = 1 - pi_1

        plabel1[i] = cdf1 * pi_1 / (cdf1 * pi_1 + cdf2 * pi_2)
        plabel2[i] = cdf2 * pi_2 / (cdf1 * pi_1 + cdf2 * pi_2)

    # Maximization
    mu1 = sum(sample * plabel1) / sum(plabel1)
    mu2 = sum(sample * plabel2) / sum(plabel2)
    sd1 = np.sqrt( sum(plabel1 * (sample - mu1) ** 2) / sum(plabel1) )
    sd2 = np.sqrt( sum(plabel2 * (sample - mu2) ** 2) / sum(plabel2) )
    pi_1 = sum(plabel1)/len(sample)
    
    newparams = np.array([mu1, mu2, pi_1, sd1, sd2])
    print(params)

    # Checking the convergence
    if np.max( abs( np.asarray(params) - np.asarray(newparams) ) ) < tol:
        converged = True

    params = newparams

plt.title('Histogram of fake data')
plt.hist(sample, bins = 100, normed = True)

x = np.linspace(sample.min(), sample.max(), 100)
plt.plot(x, mlab.normpdf(x, mu1, sd1))
plt.plot(x, mlab.normpdf(x, mu2, sd2))

plt.show()

# -------------------------------------------------------------------------------------------------------

## EXERCISE 4

# Function
def f(x, delta = 1.5):
    if x <= 0:
        return 0
    return (delta / (x * np.sqrt(2 * np.pi))) * (np.cosh(delta * np.log(2 * x))) * np.exp((-np.sinh(delta * np.log(2 * x)) ** 2) / 2)

repet = 10000
values = np.linspace(0, 5, repet)
Y = []

for i, x in enumerate(values):
    Y.append(f(x))

# plotting
plt.plot(values, Y)


####

# exp dist with lambda = 1
def f2(x):
    if x <= 0:
        return 0
    return np.exp(-x)


f_values = []
f2_values = []

# applying the method
for i, x in enumerate(values):
    f_values.append(f(x))
    f2_values.append(f2(x))

alpha = 1 / np.nanmax(np.array(f_values) / np.array(f2_values))
alpha


plt.plot(values, Y)
plt.plot(values, np.exp(-values) / alpha)

####

def rejection_sample(repet):
    samples = []
    
    for i in range(repet):
        y = np.random.exponential()
        u = np.random.uniform()
        
        criterion = f(y) / (f2(y) / alpha)
        if u <= criterion:
            samples.append(y)
            
    return np.array(samples)

# plotting
s = rejection_sample(repet)
sns.distplot(s)

EX2 = (s ** 2).mean()
EX2

estimates2 = []

for i in range(repet):
    estimates2.append((np.random.choice(s, len(s)) ** 2).mean())
    
sd = np.array(estimates2).std()
sd





