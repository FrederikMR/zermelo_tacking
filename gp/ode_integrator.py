#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:55:05 2021

@author: root
"""
#%% Sources

#https://diffeq.sciml.ai/stable/solvers/ode_solve/
#https://en.wikipedia.org/wiki/Simpson%27s_rule

#%% Modules used

import jax.numpy as jnp
from jax import lax, grad

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

from scipy.optimize import minimize

#%% Functions

def bvp_solver(p0_init, q0, qT, f_fun, t0 = 0.0, T=1.0, n_steps = 100, grid = None, max_iter=100, tol=1e-05, method='euler'):
    
    def error_fun(p0):
        
        x0 = jnp.hstack((q0,p0))
        
        qT_hat = ode_integrator(x0, f_fun, grid = grid, method='euler')[-1,:idx]
        
        return jnp.sum((qT-qT_hat)**2)
    
    if grid is None:
        grid = jnp.linspace(t0, T, n_steps)
    
    idx = len(qT)
    
    grad_error = grad(error_fun)
    
    sol = minimize(error_fun, p0_init.reshape(-1), jac=grad_error,
                 method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False
    
    #print(sol.message)
    
    p0 = sol.x
    x0 = jnp.hstack((q0, p0))
    
    xt = ode_integrator(x0, f_fun, t0, T, n_steps, grid, method)
    
    return xt

def ode_integrator(x0, f_fun, t0 = 0.0, T=1.0, n_steps = 100, grid = None, method='euler'):
        
    def euler():
        
        def step_fun(yn, time):
            
            tn, hn = time
            y = yn+f_fun(tn,yn)*hn
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
    
    def midpoint():
        
        def step_fun(yn, time):
            
            tn, hn = time
            y = yn+hn*f_fun(tn,yn)
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
    
    def heun():
        
        def step_fun(yn, time):
            
            tn, hn = time
            
            ytilden = yn+hn*f_fun(tn, yn)
            y = yn+hn/2*(f_fun(tn, yn)+f_fun(tn+hn, ytilden))
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
    
    #def ralston():
    #    
    #    def step_fun(yn, time):
    #        
    #        tn, hn = time
    #        
    #        k1 = f_fun(tn, yn)
    #        k2 = hn*f_fun(tn+2*hn/3, yn+2*k1/3)
    #        
    #        y = yn+(k1+3*k2)
    #        
    #        return y, y
    #    
    #    _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
    #    
    #    return jnp.concatenate((x0[jnp.newaxis,...],yt/4), axis=0)
    
    def bs3():
        
        def step_fun(yn, time):
            
            tn, hn = time
            hn2 = hn/2
            hn34 = 0.75*hn
            
            k1 = f_fun(tn, yn)
            k2 = f_fun(tn+hn2, yn+hn2*k1)
            k3 = f_fun(tn+hn34, yn+hn34*k2)
            
            y = yn+hn*2/9*k1+hn*k2/3+4*hn*k3/9
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
    
    def rk4():
        
        def step_fun(yn, time):
            
            tn, hn = time
            hn2 = hn/2
            
            k1 = f_fun(tn, yn)
            k2 = f_fun(tn+hn2, yn+hn2*k1)
            k3 = f_fun(tn+hn2, yn+hn2*k2)
            k4 = f_fun(tn+hn, yn+hn*k3)
            
            y = yn+hn*(k1+2*k2+2*k3+k4)
            
            return y, y
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        return jnp.concatenate((x0[jnp.newaxis,...],yt/6), axis=0)
    
    if grid is None:
        grid = jnp.linspace(t0, T, n_steps)
    
    dt_grid = jnp.hstack((jnp.diff(grid)))
    
    if method=='euler':
        return euler()
    elif method=='midpoint':
        return midpoint()
    elif method=='heun':
        return heun()
    #elif method=='ralston':
    #    return ralston()
    elif method=='bogacki-shapime':
        return bs3()
    elif method=='runge-kutta':
        return rk4()
    else:
        return euler()

def integrator(f, t0 = 0.0, T=1.0, n_steps = 100, grid = None, method='euler'):
    
    def euler():
        
        def euler_vec():
                        
            dim = jnp.ones(len(f.shape), dtype=int)
            dim = dim.at[0].set(-1)
                        
            val = f[1:]
            zero = jnp.zeros_like(f[0])[jnp.newaxis,...]

            res = jnp.cumsum(jnp.concatenate((zero, \
                                         dt_grid.reshape(dim)*val)),axis=0)
                
            return res[-1], res
        
        def euler_fun():
            
            def euler_step(carry, time):
                
                tn, hn = time
                val = carry+f(tn)*hn
                
                return val, val
                        
            yT, yt = lax.scan(euler_step, 0.0, xs=(grid[1:], dt_grid))
            
            zero = jnp.zeros_like(yT)[jnp.newaxis,...]
            
            return yT, jnp.concatenate((zero, yt), axis=0)
        
        if vec:
            return euler_vec()
        else:
            return euler_fun()
        
    def trapez():
        
        def trapez_vec():
            
            n = len(f.shape)
            if n>1:
                dim = jnp.ones(len(f.shape), dtype=int)
                dim = dim.at[0].set(-1)
            else:
                dim = -1
            
            y_right = f[1:]
            y_left = f[:-1]
            zero = jnp.zeros_like(f[0])[jnp.newaxis,...]
            
            res = jnp.concatenate((zero, \
                                jnp.cumsum(dt_grid.reshape(dim)*(y_right+y_left), axis=0)/2), \
                                  axis=0)
                
            return res[-1], res
            
        def trapez_fun():
            
            def trapez_step(carry, time):
                
                val_prev, f_prev = carry
                tn, hn = time
                
                f_up = f(tn)
                val = val_prev+(f_prev+f_up)*hn
                
                return (val, f_up), val
            
            yT, yt = lax.scan(trapez_step, (0.0, f(grid[0])), xs=(grid[1:], dt_grid))
            
            int_val = yT[0]/2
            zero = jnp.zeros_like(int_val)[jnp.newaxis,...]
            
            return int_val, jnp.concatenate((zero, yt/2), axis=0)
        
        if vec:
            return trapez_vec()
        else:
            return trapez_fun()
        
    def simpson_13():
            
        def step(carry, time):
            
            val_prev, f_prev = carry
            t_prev, tn, hn = time
            
            f_up = f(tn)
            f_mid = f((t_prev+tn)/2)
            val = val_prev+hn*(f_prev+4*f_mid+f_up)
            
            return (val, f_up), val
        
        yT, yt = lax.scan(step, (0.0, f(grid[0])), xs=(grid[:-1], grid[1:], dt_grid))
        yT = yT[0]/6

        zero = jnp.zeros_like(yT)[jnp.newaxis,...]
        
        return yT, jnp.concatenate((zero, yt/6), axis=0)
    
    #def simpson_38():
    #        
    #    def step(carry, time):
    #        
    #        val_prev, f_prev = carry
    #        t_prev, tn, hn = time
    #        
    #        f_up = f(tn)
    #        val = val_prev+hn*(f_prev+3*f((2*t_prev+tn)/3)+3*f((t_prev+2*tn)/3)+f_up)
    #        
    #        return (val, f_up), val
    #    
    #    yT, yt = lax.scan(step, (0.0, f(grid[0])), xs=(grid[:-1], grid[1:], dt_grid))
    #    yT = yT[0]*3/8
    #    
    #    zero = jnp.zeros_like(yT)[jnp.newaxis,...]
    #    
    #    return yT, jnp.concatenate((zero, yt*3/8), axis=0)

    if grid is None:
        grid = jnp.linspace(t0, T, n_steps)
    
    if callable(f):
        vec = False
    else:
        vec = True
    
    dt_grid = jnp.diff(grid)
    
    if vec:
        if method=='euler':
            return euler()
        elif method=='trapez':
            return trapez()
        else:
            return euler()
    else:
        if method=='euler':
            return euler()
        elif method=='trapez':
            return trapez()
        elif method=='simpson-13':
            return simpson_13()
        #elif method=='simpson-38':
        #    return simpson_38()
        else:
            return euler()
    