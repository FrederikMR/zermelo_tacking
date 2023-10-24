#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:50:38 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JAXGeometry.setup import *
from JAXGeometry.params import *

#%% Riemannian Newton's Method

@jit
def RMNewtonsMethod(mu_init:ndarray,
                    M:object,   
                    grad_fn:Callable[[Tuple[ndarray, ndarray]], ndarray],
                    ggrad_fn:Callable[[Tuple[ndarray, ndarray]], ndarray] = None,
                    step_size:float = 0.1,
                    max_iter:int=100,
                    bnds:tuple[ndarray, ndarray]=(None, None),
                    max_step:ndarray=None
                    )->Tuple[Tuple[ndarray, ndarray], ndarray]:
    
    @jit
    def update(carry:Tuple[ndarray, ndarray], idx:int
               )->Tuple[Tuple[ndarray, ndarray],
                        Tuple[ndarray, ndarray]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu = M.Exp(mu, -step_size*step_grad)
        mu[0] = jnp.clip(mu[0], lb, ub)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
        
    if ggrad_fn is None:
        ggrad_fn = jacfwdx(grad_fn)
        
    grad = grad_fn_rm(mu_init)
    _, out = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))
        
    mu = out[0]
    grad = out[1]

    return mu, grad

#%% Euclidean Newton's Method

@jit
def NewtonsMethod(mu_init:ndarray,
                  grad_fn:Callable[[ndarray], ndarray],
                  ggrad_fn:Callable[[Tuple[ndarray, ndarray]], ndarray] = None,
                  step_size:float = 0.1,
                  max_iter:int=100,
                  bnds:Tuple[ndarray, ndarray]=(None,None),
                  max_step:ndarray=None
                  )->Tuple[ndarray, ndarray]:
    
    @jit
    def update(carry:Tuple[ndarray, ndarray], idx:int
               )->Tuple[Tuple[ndarray, ndarray],
                        Tuple[ndarray, ndarray]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu -= step_size*step_grad
        mu = jnp.clip(mu, lb, ub)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb = bnds[0]
    ub = bnds[1]
        
    if ggrad_fn is None:
        ggrad_fn = jacfwdx(grad_fn)
        
    grad = grad_fn_rm(mu_init)
    _, out = lax.scan(update, init=(mu_init, grad), xs=jnp.arange(0,max_iter,1))
        
    mu = out[0]
    grad = out[1]

    return mu, grad

#%% Joint Newton's Method

@jit 
def JointNewtonsMethod(mu_rm:ndarray,
                       mu_euc:ndarray,
                       M:object,
                       grad_fn_rm:Callable[[Tuple[ndarray, ndarray]], ndarray],
                       grad_fn_euc:Callable[[ndarray], ndarray],
                       ggrad_fn_rm:Callable[[Tuple[ndarray, ndarray]], ndarray] = None,
                       ggrad_fn_euc:Callable[[ndarray], ndarray] = None,
                       step_size_rm:float = 0.1,
                       step_size_euc:float = 0.1,
                       max_iter:int=100,
                       bnds_rm:Tuple[ndarray, ndarray]=(None,None),
                       bnds_euc:Tuple[ndarray, ndarray]=(None,None),
                       max_step:ndarray=None
                       )->Tuple[Tuple[ndarray, ndarray], ndarray, ndarray, ndarray]:
    
    @jit
    def update(carry:Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], idx:int
               )->Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc = carry
        
        ggrad_rm = ggrad_fn_rm(mu_rm)
        ggrad_euc = ggrad_fn_euc(mu_euc)
        
        step_grad_rm = jscipy.sparse.linalg.gmres(ggrad_rm, grad_rm)[0]
        step_grad_euc = jscipy.sparse.linalg.gmres(ggrad_euc, grad_euc)[0]
        
        step_grad_rm = jnp.clip(step_grad_rm, min_step, max_step)
        step_grad_euc = jnp.clip(step_grad_euc, min_step, max_step)
        
        mu_rm = M.Exp(mu_rm, -step_size_rm*step_grad_rm)
        mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        
        mu_euc -= step_size_euc*step_grad_euc
        mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
        
        new_chart = M.centered_chart(mu_rm)
        mu_rm = M.update_coords(mu_rm,new_chart)
        
        grad_rm = grad_fn_rm(mu_rm)
        grad_euc = grad_fn_euc(mu_euc)
        out = (mu_rm, mu_euc, grad_rm, grad_euc)
    
        return out, out
    
    if max_step is None:
        min_step = None
    else:
        min_step = -max_step
    lb_euc = bnds_euc[0]
    ub_euc = bnds_euc[1]
    lb_rm = bnds_rm[0]
    ub_rm = bnds_rm[1]
        
    if ggrad_fn_rm is None:
        ggrad_fn_rm = jacfwdx(grad_fn_rm)
        
    if ggrad_fn_euc is None:
        ggrad_fn_euc = jacfwdx(grad_fn_euc)
        
    grad_rm = grad_fn_rm(mu_rm)
    grad_euc = grad_fn_euc(mu_euc)
    _, out = lax.scan(update, init=(mu_rm, mu_euc, grad_rm, grad_euc), xs=jnp.arange(0,max_iter,1))
        
    mu_rm = out[0]
    mu_euc = out[1]
    grad_rm = out[2]
    grad_euc = out[3]
    
    return mu_rm, mu_euc, grad_rm, grad_euc








