#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:22:38 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import vmap, grad, jacfwd, jacrev, value_and_grad
from jax import lax
from jax import random as jrandom

import jax.numpy as jnp

import jax 
jax.config.update("jax_enable_x64", True)

#jac scipy
import jax.scipy as jscipy
from jax.scipy.optimize import minimize as jminimize

#JAX Optimization
from jax.example_libraries import optimizers

#scipy
from scipy.optimize import minimize

from abc import ABC
from typing import Callable, Tuple, Dict, List

#%% Class

class SDESampler(ABC):
    def __init__(self,
                 drift_fun:Callable,
                 diffusion_fun:Callable,
                 dt_steps:int=1000,
                 seed:int=2712,
                 )->None:
        
        self.drift_fun = drift_fun
        self.diffusion_fun = diffusion_fun
        
        self.dt_steps = dt_steps
        self.seed = seed
        self.key = jrandom.key(seed)
        
        self.dtype = None
        
        return
    
    def W_sample(self,
                  dt:Array,
                  n_samples:int=1,
                  )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        
        z = jrandom.normal(subkey, shape=(self.dt_steps, n_samples, self.dim))
        
        return z.squeeze()

    def dX(self, 
           z:Array, 
           step:Tuple[Array,Array,Array],
           )->Tuple[Array,Array]:
        
        dt, t, dW = step
        t += dt

        if z.ndim == 1:
            drift_term = self.drift_fun(t,z)
            diffusion_term = self.diffusion_fun(t,z)
        else:
            drift_term = vmap(self.drift_fun, in_axes=(None,0))(t,z)
            diffusion_term = vmap(self.diffusion_fun, in_axes=(None,0))(t,z)
            
        det = (drift_term*dt.reshape(-1,1)).squeeze()
        stoch = jnp.einsum('...ik,...k->...i', diffusion_term, dW)
        
        z += det+stoch
        z = z.astype(self.dtype)
        
        return (z,)*2
    
    def __call__(self,
                 z0:Array,
                 T:Array=1.0,
                 )->Tuple[Array,Array,Array,Array]:
        
        self.dtype = z0.dtype
        
        if z0.ndim == 1:
            n_samples = 1
        else:
            n_samples = len(z0)
            
        dt = jnp.array([T/self.dt_steps]*self.dt_steps)
        t = jnp.cumsum(dt)
        dW = jnp.einsum('i,i...j->i...j', 
                        jnp.sqrt(dt), 
                        self.W_sample(dt, n_samples=n_samples)
                        )

        _, xt = lax.scan(self.dX,
                         init = z0,
                         xs = (dt,t,dW),
                         )
        
        return xt
    
#%% Diffusion Time

class DiffusionTime(SDESampler):
    def __init__(self,
                 drift_fun:Callable,
                 diffusion_fun:Callable,
                 dim:int = 1,
                 dt_steps:int=100,
                 seed:int=2712,
                 max_iter:int=100,
                 )->None:
        super().__init__(drift_fun, diffusion_fun, dt_steps, seed)
        
        self.drift_fun = drift_fun
        self.diffusion_fun = diffusion_fun
        self.dim = dim
        self.max_iter = max_iter
        
        self.M_fun = lambda t,*theta: self.trapez_rule(lambda t: self.drift_fun(t,*theta), jnp.linspace(0,t,dt_steps))
        self.S_fun = lambda t, *theta: .5*self.trapez_rule(lambda t: self.diffusion_fun(t,*theta)**2, jnp.linspace(0,t,dt_steps))
            
        return
    
    def __str__(self)->str:
        
        return "Time-only dependent drift and diffusion Ito process"
    
    def trapez_rule(self, f_fun:Callable, t:Array)->Array:
        
        dt = jnp.diff(t)
        if callable(f_fun):
            fval = vmap(f_fun)(t).squeeze()
        else:
            fval = f_fun
    
        return jnp.concatenate((jnp.zeros((1, self.dim)), 
                                .5*jnp.cumsum((dt*(fval[1:]+fval[:-1]).T).T.reshape(-1, self.dim), axis=0)),
                               axis=0)
        
    def mu(self, x0:Array, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, x0+self.trapez_rule(lambda t: self.drift_fun(t,*theta), t)
    
    def var(self, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, self.trapez_rule(lambda t: self.diffusion_fun(t,*theta)**2, t)
        
    def pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St))
        else:
            val = vmap(lambda Mt, St, x0: jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St)))(Mt, St, x0)
            
            return jnp.prod(val)
        
    def log_pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def max_likelihood(self, X:Array, t:Array, x0:Array, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: vmap(lambda x,t: self.log_pdf(x,x0,t,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    def integral_mu(self, w0:Array, x0_fun:Callable, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        return Mt+x0+w0*T
    
    def integral_var(self, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        
        return St
    
    def integral_log_pdf(self, x:Array, x0_fun:Callable, t:Array, dim:int=2, *theta)->Array:
        
        t_grid = jnp.linspace(0,t,self.n_points).squeeze()        
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        if self.dim == 1:
            return -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def integral_max_likelihood(self, X:Array, t:Array, x0:Callable, dim, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: 
                                  vmap(lambda x,t: self.integral_log_pdf(x,x0, t,dim,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x