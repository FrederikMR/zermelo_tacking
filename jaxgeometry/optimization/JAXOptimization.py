#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:04:19 2023

@author: fmry
"""

#%% Sources

#%% Modules

from jaxgeometry.setup import *

#%% Riemannian Jax Optimization

def RMJaxOpt(mu_init:ndarray,
             M:object,
             grad_fn:Callable[[Tuple[ndarray, ndarray]], ndarray],
             max_iter:int=100,
             optimizer:Callable=None,
             opt_params:Tuple=(0.1, 0.9, 0.999, 1e-8),
             bnds:Tuple[ndarray, ndarray]=(None, None),
             max_step:ndarray=None
             )->Tuple[ndarray, ndarray]:
    
    @jit
    def update(carry:Tuple[ndarray, ndarray, object], idx:int
               )->Tuple[Tuple[ndarray, ndarray, object],
                        Tuple[ndarray, ndarray]]:
        
        mu, grad, opt_state = carry
        
        #grad = jnp.clip(grad, min_step, max_step)
        
        opt_state = opt_update(idx, grad, opt_state)
        mu_rm = get_params(opt_state)
        #mu_rm = jnp.clip(mu_rm, lb, ub)
        
        new_chart = M.centered_chart((mu_rm, mu[1]))
        mu = M.update_coords((mu_rm, mu[1]),new_chart)
        
        grad = grad_fn(mu)
        
        return (mu, grad, opt_state), (mu, grad)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(mu_init[0])
    grad = grad_fn(mu_init)
    _, out = scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    mu = out[0]
    grad = out[1]
    
    return mu, grad

#%% Euclidean Jax Optimization

def JaxOpt(mu_init:ndarray,
           M:object,
           grad_fn:Callable[[ndarray], ndarray],
           max_iter:int=100,
           optimizer:Callable=None,
           opt_params:Tuple=(0.1, 0.9, 0.999, 1e-8),
           bnds:Tuple[ndarray, ndarray]=(None,None),
           max_step=None
           )->Tuple[ndarray, ndarray]:
    
    @jit
    def update(carry:Tuples[ndarray, ndarray, object], idx:int
               )->Tuple[Tuple[ndarray, ndarray, object],
                        Tuple[ndarray, ndarray]]:
        
        mu, grad, opt_state = carry
        
        #grad = jnp.clip(grad, min_step, max_step)
        
        opt_state = opt_update(idx, grad, opt_state)
        mu = get_params(opt_state)
        
        mu = jnp.clip(mu, lb, ub)
        
        grad = grad_fn(mu)
        
        return (mu, grad, opt_state), (mu, grad)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(mu_init)
    grad = grad_fn(mu_init)
    _, out = scan(update, init = (mu_init, grad, opt_state), xs = jnp.arange(0,max_iter,1))
    mu = out[0]
    grad = out[1]
    
    return mu, grad

#%% Joint Jax Optimization

def JointJaxOpt(mu_rm:ndarray,
                mu_euc:ndarray,
                M:object,
                grad_fn_rm:Callable[[Tuple[ndarray, ndarray]], ndarray],
                grad_fn_euc:Callable[[ndarray], ndarray],
                max_iter:int=100,
                optimizer:Callable=None,
                opt_params:Tuple=(0.1, 0.9, 0.999, 1e-8),
                bnds_rm:Tuple[ndarray, ndarray]=(None,None),
                bnds_euc:Tuple[ndarray, ndarray]=(None,None),
                max_step:jnp.ndarray=None
                )->Tuple[ndarray, ndarray, ndarray, ndarray]:
    
    @jit
    def update(carry:Tuple[ndarray, ndarray, ndarray, ndarray, object], 
               idx:int
               )->Tuple[Tuple[ndarray, ndarray, ndarray, ndarray, object],
                        Tuple[ndarray, ndarray, ndarray, ndarray]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc, opt_state = carry
        
        grad_rm = grad_rm #jnp.clip(grad_rm, min_step, max_step)
        grad_euc = grad_euc #jnp.clip(grad_euc, min_step, max_step)
        
        grad = jnp.hstack((grad_rm, grad_euc))
        opt_state = opt_update(idx, grad, opt_state)
        mu = get_params(opt_state)
        
        mux_rm = mu[:N_rm]
        mu_euc = mu[N_rm:]
        
        mux_rm = mux_rm#jnp.clip(mux_rm, lb_rm, ub_rm)
        mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
        
        new_chart = M.centered_chart((mux_rm, mu_rm[1]))
        mu_rm = M.update_coords((mux_rm, mu_rm[1]),new_chart)
        
        grad_rm = grad_fn_rm(mu_rm, mu_euc)
        grad_euc = grad_fn_euc(mu_rm, mu_euc)
        
        return (mu_rm, mu_euc, grad_rm, grad_euc, opt_state), \
            (mu_rm, mu_euc, grad_rm, grad_euc)
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb_euc = bnds_euc[0]
    ub_euc = bnds_euc[1]
    lb_rm = bnds_rm[0]
    ub_rm = bnds_rm[1]
    
    if optimizer is None:
        optimizer = optimizers.adam
        opt_init, opt_update, get_params = optimizer(0.1, b1=0.9, b2=0.999, eps=1e-8)
    else:
        opt_init, opt_update, get_params = optimizer(*opt_params)
        
    opt_state = opt_init(jnp.hstack((mu_rm[0], mu_euc)))
    grad_rm = grad_fn_rm(mu_rm, mu_euc)
    grad_euc = grad_fn_euc(mu_rm, mu_euc)
    N_rm = len(grad_rm)
    _, out = scan(update, init = (mu_rm, mu_euc, 
                                      grad_rm, grad_euc, 
                                      opt_state), xs = jnp.arange(0,max_iter,1))
    mu_rm = out[0]
    mu_euc = out[1]
    grad_rm = out[2]
    grad_euc = out[3]
    
    return mu_rm, mu_euc, grad_rm, grad_euc

