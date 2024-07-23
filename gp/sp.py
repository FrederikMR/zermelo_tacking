#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:03:15 2022

@author: frederik
"""

#%% Modules

#JAX
import jax.numpy as jnp
from jax import lax, vmap, random

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)


#%% Initilization

global key, seed_step
seed = 2712
seed_step = 1
key = random.PRNGKey(seed)

#%% Functions

def mnormal_pdf(X, mu, cov):
    cov_inv = jnp.linalg.inv(cov)
    det = jnp.linalg.det(cov)
    k = X.shape[-1]
    den = jnp.sqrt(det*(2*jnp.pi)**k)
    def pdf_step(x,mu):
        
        x_diff = x-mu
        val = jnp.exp(-1/2*x_diff.T.dot(cov_inv).dot(x_diff))        
        
        return val/den
    
    if len(X.shape)==1:
        return pdf_step(X,mu)
    else:
        return vmap(pdf_step)(X,mu)
    
def sim_unif(a:float=0.0, b:float=1.0, dim:int=1):
    
    global key
    
    keys = random.split(key,num=seed_step+1)
    key = keys[0]
    subkeys = keys[1:]
    
    if dim==1:
        U = random.uniform(subkeys[0], minval=a, maxval=b)
    else:
        U = random.uniform(subkeys[0], minval=a, maxval=b, shape=[dim])
    
    return U

def sim_multinormal(mu:jnp.ndarray=jnp.zeros(2), cov:jnp.ndarray=jnp.eye(2), dim:int=1):
    
    global key
    
    keys = random.split(key,num=seed_step+1)
    key = keys[0]
    subkeys = keys[1:]
    
    if dim==1:
        Z = random.multivariate_normal(subkeys[0], mean=mu,
                               cov=cov)
    else:
        Z = random.multivariate_normal(subkeys[0], mean=mu,
                               cov=cov, shape=dim)
    
    return Z

def sim_normal(mu=0.0, sigma=1.0, simulations=1):
    
    global key
    
    keys = random.split(key,num=seed_step+1)
    key = keys[0]
    subkeys = keys[1:]
    
    Z = mu+sigma*random.normal(subkeys[0], shape=[simulations])
    
    return Z

def sim_Wt(grid, dim:int=1, simulations:int=1):
    
    global key
    keys = random.split(key,num=seed_step+1)
    key = keys[0]
    subkeys = keys[1:]
    
    n_steps = len(grid)
    sqrtdt = jnp.sqrt(jnp.diff(grid, axis=0)).reshape(-1,1)
    N = random.normal(subkeys[0],[simulations, n_steps-1, dim])
    
    Wt = jnp.zeros([simulations, n_steps, dim])
    Wt = Wt.at[:,1:].set(sqrtdt*N)
        
    return jnp.cumsum(Wt, axis=1).squeeze()

def sim_dWt(grid, dim:int=1):
    
    global key
    keys = random.split(key,num=seed_step+1)
    key = keys[0]
    subkeys = keys[1:]
    
    n_steps = len(grid)
    sqrtdt = jnp.sqrt(jnp.diff(grid, axis=0)).reshape(-1,1)
    N = random.normal(subkeys[0],[n_steps-1, dim])
        
    return grid, (sqrtdt*N).squeeze()

def sim_sde_euler(x0:jnp.ndarray, 
                  b_fun,
                  sigma_fun,
                  Wt:jnp.ndarray,
                  grid,
                  args=()):
    
    def sde_step(yi, ite):
        
        t, dt, dWt = ite
        y = yi+b_fun(t, yi)*dt+jnp.dot(sigma_fun(t, yi),dWt)
        
        return y, y

    if args:
        b_fun = lambda t,x : b_fun(t,x, *args)
        sigma_fun = lambda t,x: sigma_fun(t,x,*args)
    
    diff_t = jnp.diff(grid)
    dW = jnp.diff(Wt, axis=0)

    _, y = lax.scan(sde_step, x0, xs=(grid[:-1], diff_t, dW))
    
    return jnp.concatenate((x0.reshape(1,-1), y), axis=0).squeeze()