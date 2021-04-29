#!/usr/bin/env python
# coding: utf-8

# In[39]:

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.stats import norm, laplace
import random
import math
global pdf
import scipy.optimize as optimize


# ## Task 2.1 and 2.2 - task 1.1-1.2 (faulty measurements)

# Probability = $\gamma$ the sensor returns something meaningless  
# Prob = 1 -  $\gamma$ the sensor works fine

# In[42]:


def arbitrary_poly(params):
    poly_model = lambda x: sum([p*(x**i) for i, p in enumerate(params)])
    return poly_model

# params: [theta_0, theta_1, theta_2]
true_params = [3,6,2]
y_model = arbitrary_poly(true_params)

# Plot
x = np.linspace(start=-3, stop=3, num=20)
plt.figure()
plt.plot(x, y_model(x))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model")


# In[43]:


# Hyperparameters for the type of noise-generating distribution.
loc = 0           # location (mean) parameter 
scale = 1         # scaling (std dev) parameter
magnitude = 1.2   # noise magnitude
N = 201           # number of samples

# Generate data points
range_low, range_high = -3, 3
u = np.sort(np.random.uniform(range_low,range_high,N))
y_true = y_model(u)

# Generate noise
y = y_true
probability = 0
laplaceBeta = 1
normVariance = 1
gamma = 0.2

noiseLaplace = magnitude * np.random.laplace(loc, laplaceBeta, int((1-probability)*N))

for i in range(0, N):
    if (np.random.binomial(1, gamma)):
        y[i] = 100
    else:
        y[i] = y_true[i] + noiseLaplace[i]

# Plot measured data
plt.scatter(u, y, label=r"Measured data")
u0 = np.linspace(min(u), max(u), N)
plt.plot(u0, y_model(u0), "k", alpha=0.3, lw=3, label="True model")
plt.legend()
plt.xlabel("x")
plt.ylabel("y");


# ## Task 2.3

# ## LS:

# In[69]:


def LS_func(order, u, y, N):
    # Matrix form
    u_tensor_0 = np.reshape(u,(N,1))

    ones_vec = np.ones((N,1))
    u_tensor = ones_vec

    for i in range(1, order):
        u_tensor = np.append(u_tensor, np.power(u_tensor_0, i) ,axis=1)

    # Step 2 - calculate inverse u^T*u
    u_transpose_dot_u = np.dot(u_tensor.T,u_tensor)  # calculating dot product
    u_transpose_dot_u_inv = np.linalg.inv(u_transpose_dot_u) #calculating inverse

    # Step 3 - calculate u^T*u
    u_transpose_dot_y = np.dot(u_tensor.T,y)  # calculating dot product
    
    # Step 4
    LS_params = np.dot(u_transpose_dot_u_inv,u_transpose_dot_y)
    LS_params_rounded = ["{:.2f}".format(round(i, 2)) for i in LS_params.tolist()]
    
    diffParams = []
    for i in range(0, order):
        diffParams.append(float(true_params[i] - float(LS_params_rounded[i])))
    
    LS_params = LS_params.tolist()
    LS_estimate = arbitrary_poly(LS_params)
    return LS_params, LS_estimate

LS_params_1, LS_estimate_1 = LS_func(1, u, y, N)
LS_params_2, LS_estimate_2 = LS_func(2, u, y, N)
LS_params_3, LS_estimate_3 = LS_func(3, u, y, N)

# Plot the true vs estimated model
plt.scatter(u, y, label=r"Measured data $\mathcal{N}(\mu, \sigma)$")
u0 = np.linspace(min(u), max(u), N)
plt.plot(u0, y_model(u0), "r", alpha=0.7, lw=3, label="True model")
plt.plot(u0, LS_estimate_1(u0), color="black", linestyle="--", lw=3, label="LS estimate, order 1")
plt.plot(u0, LS_estimate_2(u0), color="pink", linestyle="--", lw=3, label="LS estimate, order 2")
plt.plot(u0, LS_estimate_3(u0), color="yellow", linestyle="--", lw=3, label="LS estimate, order 3")
#plt.xlim(0, 10)
plt.legend()
plt.xlabel("x")
plt.ylabel("y");


# ## ML:

# ### Func for calculating the log likelihood function

# In[48]:


y = y_true

# Step 1 - define the log likelihood function to be minimized

def log_likFunction(par_vec, y, x):
    # Use the distribution class chosen earlier
    # If the standard deviation parameter is negative, return a large value:
    if par_vec[-1] < 0:
        return(1e8)
    # The likelihood function values:
    lik = pdf(y,
              loc = sum([p*(x**i) for i, p in enumerate(par_vec[:-1])]),
              scale = par_vec[-1])
    
    #This is similar to calculating the likelihood for Y - XB
    # res = y - par_vec[0] - par_vec[1] * x
    # lik = norm.pdf(res, loc = 0, sd = par_vec[2])
    
    # If all logarithms are zero, return a large value
    if all(v == 0 for v in lik):
        return(1e8)
    # Logarithm of zero = -Inf
    return(-sum(np.log(lik[np.nonzero(lik)])))


# ### Func for calculating MLE:

# In[51]:


y = y_true
#pdf = laplace.pdf
def MLEFunction(order, u, y, N):

    # The likelihood function includes the scale (std dev) parameter which is also estimated by the optimized
    # therefore the initial guess verctor has length n+2 [theta_0_hat, theta_1_hat, ... , theta_n_hat, sigma_hat]
    init_guess = np.zeros(order + 1)
    init_guess[-1] = N

    # Do Maximum Likelihood Estimation:
    opt_res = optimize.minimize(fun = log_likFunction,
                                x0 = init_guess,
                                options = {'disp': True},
                                args = (y, u))

    MLE_params = opt_res.x[:-1]
    MLE_estimate = arbitrary_poly(MLE_params)

    MLE_params_rounded = ["{:.2f}".format(round(i, 2)) for i in MLE_params.tolist()]
    return MLE_params, MLE_estimate


# ### Calculating ML for different orders:

# In[55]:


ML_params_1, ML_estimate_1 = MLEFunction(1, u, y, N)
ML_params_2, ML_estimate_2 = MLEFunction(2, u, y, N)
ML_params_3, ML_estimate_3 = MLEFunction(3, u, y, N)

# print(f"\nTrue model parameters: {true_params}")

# Plot measured data
plt.scatter(u, y, label=r"Measured data $\mathcal{N}(\mu, \sigma)$")
u0 = np.linspace(min(u), max(u), N)
plt.plot(u0, y_model(u0), "r", alpha=0.5, lw=3, label="True model")
plt.plot(u0, ML_estimate_1(u0), color="black", linestyle="--", alpha=0.5, lw=3, label="ML estimate, order 1")
plt.plot(u0, ML_estimate_2(u0), color="red", linestyle="--", alpha=0.7, lw=3, label="ML estimate, order 2")
plt.plot(u0, ML_estimate_3(u0), color="yellow", linestyle="--", alpha=1, lw=3, label="ML estimate, order 3")
plt.legend()
plt.xlabel("x")
plt.ylabel("y");


# ## Task 2.4

# Code the full "model order selection / parameters estimation / predictions performance estimation" paradigm by considering the first third of the dataset as a training set, the second as a test set, and the third as a validation set. As "prediction performance index" consider the sum of the absolute deviations between the actually measured $y_t$'s and the predicted ones $\widehat{y}_t$ (i.e., $\widehat{y}_t = \left[ \widehat{\theta}_0, \widehat{\theta}_1, \widehat{\theta}_2, \ldots \right] \left[ 1, u_t, u_t^2, \ldots \right]^T$ with the $\widehat{\theta}$'s the estimated parameters corresponding to the selected model order). Use the same performance index to solve also the model order selection problem
# 

# In[79]:


y = y_true
def getModels(order, u, y):
    LS_params = []
    LS_estimates = []
    MLE_params = []
    MLE_estimates = []
    N = len(u)
    
    for i  in range(order):
        tempParams, tempEstims = LS_func(i+1, u, y, N)
        LS_params.append(tempParams)
        LS_estimates.append(tempEstims)
        tempParams, tempEstims = MLEFunction(i+1, u, y, N)
        MLE_params.append(tempParams)
        MLE_estimates.append(tempEstims)
    return LS_params, LS_estimates, MLE_params, MLE_estimates
    
def createNewSets(u, y):
    noise_xy = np.array([u, y]).T
    np.random.shuffle(noise_xy)
    
    training_set = noise_xy[::3]
    training_set = training_set[training_set[:,0].argsort()].T
    
    test_set = noise_xy[1::3]
    test_set = test_set[test_set[:,0].argsort()].T
    
    validation_set = noise_xy[2::3]
    validation_set = validation_set[validation_set[:,0].argsort()].T
    
    true_training = y_model(training_set[0])
    true_test = y_model(test_set[0])
    validation_set = y_model(validation_set[0])
    return training_set, test_set, validation_set

training_set, test_set, validation_set = createNewSets(u, y)

true_training = y_model(training_set[0])
true_test = y_model(test_set[0])
validation_set = y_model(validation_set[0])

def singlePerf(y_t, y_hat):
    return sum(abs(y_t - y_hat))

def modelSelect(performance):
    return performance.index(min(performance))

def modelsPerf(order, LS_estimates, MLE_estimates, validation_set, test_set):
    LS_valid_dev = []
    MLE_valid_dev = []
    
    for i in range(order):
        tmpLS = singlePerf(validation_set[1], LS_estimates[i](validation_set[0]))
        LS_valid_dev.append(tmpLS)
        tmpMLE = singlePerf(validation_set[1], MLE_estimates[i](validation_set[0]))
        MLE_valid_dev.append(tmpMLE)
        
    LS_opt_ind = modelSelect(LS_valid_dev)
    LS_opt_model = LS_estimates[LS_opt_ind]
    
    MLE_opt_ind = modelSelect(MLE_valid_dev)
    MLE_opt_model = MLE_estimates[MLE_opt_ind]
    #test
    #print(f"Best LS model: {LS_opt_ind+1}")
    #print(f"Best ML model: {MLE_opt_ind+1}")
    LS_perf_test = singlePerf(test_set[1], LS_opt_model(test_set[0]))
    MLE_perf_test = singlePerf(test_set[1], MLE_opt_model(test_set[0]))
    #print(f"\nLS_model_{LS_opt_ind + 1}_dev_test: {LS_perf_test}")
    #print(f"ML_model_{MLE_opt_ind + 1}_dev_test: {MLE_perf_test}")

    return LS_opt_model, MLE_opt_model, LS_perf_test, MLE_perf_test


LS_params, LS_estimates, MLE_params, MLE_estimates= getModels(3, training_set[0], training_set[1])
LS_opt_model, MLE_opt_model, LS_perf_test, MLE_per_test = modelsPerf(3, LS_estimates, MLE_estimates, validation_set, test_set)
plt.figure(1)
plt.scatter(validation_set[0],validation_set[1] ,  label="True Model")

plt.plot(validation_set[0], LS_estimates[0](validation_set[0]), color="darkorange" ,linestyle="--",  lw=3, label="LS estimate, order 1")
plt.plot(validation_set[0], LS_estimates[1](validation_set[0]), color="black" ,linestyle="--",  lw=3, label="LS estimate, order 2")
#plt.scatter(validation_set[0], LS_estimates[2](validation_set[0]))
plt.plot(validation_set[0], LS_estimates[2](validation_set[0]), color="green", linestyle='--', lw=3, label="LS estimate, order 3")
#plt.xlim(0, 10)
plt.legend()
plt.title("LS")
plt.xlabel("x")
plt.ylabel("y")

plt.figure(2)
plt.scatter(validation_set[0],validation_set[1] ,  label="True Model")
plt.title("MLE")

plt.plot(validation_set[0], MLE_estimates[0](validation_set[0]), color="darkorange" ,linestyle="--",  lw=3, label="MLE estimate, order 1")
plt.plot(validation_set[0], MLE_estimates[1](validation_set[0]), color="black" ,linestyle="--",  lw=3, label="MLE estimate, order 2")
#plt.scatter(validation_set[0], LS_estimates[2](validation_set[0]))
plt.plot(validation_set[0], MLE_estimates[2](validation_set[0]), color="green", linestyle='--', lw=3, label="MLE estimate, order 3")
#plt.xlim(0, 10)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")






