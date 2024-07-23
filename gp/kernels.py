#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 22:46:31 2022

@author: frederik
"""

#%% Sources

#https://www.longdom.org/open-access/a-complete-list-of-kernels-used-in-support-vector-machines-2167-0501-1000195.pdf

#%% Modules

import jax.numpy as jnp
from jax import vmap

#%% Kernels

def polynomial_kernel(x,y, d=2, c=0.0):
    
    return (x.dot(y)+c)**d

def gaussian_kernel(x,y, beta = 1.0, omega=1.0):
    
    x_diff = x-y
    
    return (beta**2)*jnp.exp(-(omega**2)*jnp.dot(x_diff,x_diff)/2)

def sigmoid_kernel(x, y, kappa=1.0, c=-0.1):
    
    return jnp.tanh(kappa*x.dot(y)+c)

def linear_kernel(x,y, c=0.0):
    
    return x.dot(y)+c

def cosine_kernel(x,y):
    
    return x.dot(y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y))

def multiquadratic_kernel(x,y, c=1.0):
    
    x_diff = x-y
    
    return jnp.sqrt(jnp.dot(x_diff, x_diff)+c**2)

def log_kernel(x,y, d=2):
    
    return -jnp.log(x-y)**d+1

def cauchy_kernel(x,y, sigma=1.0):
    
    x_diff = x-y
    
    return 1/(1+jnp.dot(x_diff, x_diff)/sigma)

def thin_plate_kernel(x,y):
    
    n = len(x)
    
    return jnp.linalg.norm(x-y)**(2*n+1)

#%% General kernel computations

def km(X, Y=None, kernel_fun=None):
    
    if Y is None:
        Y = X
    if kernel_fun is None:
        kernel_fun = gaussian_kernel
    
    #Kernel matrix
    return vmap(lambda x: vmap(lambda y: kernel_fun(x,y))(Y))(X)