#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:55:46 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JAXGeometry.setup import *
from JAXGeometry.params import *

#%% Riemannian Gradient-Newton Method

@jit
def RMGradientNewton(mu_init:jnp.ndarray, 
                     M:object,
                     grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
                     ggrad_fn:Callable[[jnp.ndarray], jnp.ndarray] = None,
                     grad_step:float = 0.1,
                     newton_step:float = 0.1,
                     iter_step:int = 10,
                     tol = 1e-1,
                     max_iter:int=100,
                     bnds:Tuple[jnp.ndarray, jnp.ndarray] = (None, None),
                     max_step:jnp.ndarray=None
                     )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update_gradient(carry:Tuple[jnp.ndarray, jnp.ndarray], idx:int
                        )->Tuple[Tuple[jnp.ndarray, jnp.ndarray],
                                 Tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        mu = M.Exp(mu, -grad_step*grad)
        mu[0] = jnp.clip(mu[0], lb, ub)
        
        new_chart = M.centered_chart(mu)
        mu = M.update_coords(mu,new_chart)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    @jit
    def update_newton(carry:Tuple[jnp.ndarray, jnp.ndarray], idx:int
                      )->Tuple[Tuple[jnp.ndarray, jnp.ndarray],
                               Tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu = M.Exp(mu, -newton_step*step_grad)
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
        
    step_grid = jnp.arange(0, iter_step, 1)
    mu_lst = []
    grad_lst = []    
    grad = [grad_fn(mu)]
    mu = [mu_init]
    for i in range(max_iter):
        _, out = lax.scan(update_gradient, init=(mu[-1], grad[-1]), xs=step_grid)
        mu = out[0]
        grad = out[1]
        mu_lst.append(mu)
        grad_lst.append(grad)
        if jnp.linalg.norm(grad) < tol:
            _, out = lax.scan(update_newton, init=(mu[-1], grad[-1]), xs=step_grid)
            mu = out[0]
            grad = out[1]
            mu_lst.append(mu)
            grad_lst.append(grad)
            
    mu = jnp.stack(mu_lst)
    grad = jnp.stack(grad_lst)
    
    return mu, grad

#%% Euclidean Gradient-Newton Method

@jit
def GradientNewton(mu_init:jnp.ndarray, 
                   grad_fn:Callable[[jnp.ndarray], jnp.ndarray],
                   ggrad_fn:Callable[[jnp.ndarray], jnp.ndarray] = None,
                   grad_step:float = 0.1,
                   newton_step:float = 0.1,
                   iter_step:int = 10,
                   tol = 1e-1,
                   max_iter:int=100,
                   bnds:Tuple[jnp.ndarray, jnp.ndarray]=(None,None),
                   max_step:jnp.ndarray=None
                   )->Tuple[jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update_gradient(carry:Tuple[jnp.ndarray, jnp.ndarray], idx:int
                        )->Tuple[Tuple[jnp.ndarray, jnp.ndarray],
                                 Tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad = carry
        
        grad = jnp.clip(grad, min_step, max_step)
        mu -= grad_step*grad
        mu = jnp.clip(mu, lb, ub)
        
        grad = grad_fn(mu)
        out = (mu, grad)
    
        return out, out
    
    @jit
    def update_newton(carry:Tuple[jnp.ndarray, jnp.ndarray], idx:int
                      )->Tuple[Tuple[jnp.ndarray, jnp.ndarray],
                               Tuple[jnp.ndarray, jnp.ndarray]]:
        
        mu, grad = carry
        
        ggrad = ggrad_fn(mu)
        
        step_grad = jscipy.sparse.linalg.gmres(ggrad, grad)[0]
        step_grad = jnp.clip(step_grad, min_step, max_step)
        
        mu -= newton_step*step_grad
        mu = jnp.clip(mu, min_step, max_step)
        
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
        
    step_grid = jnp.arange(0, iter_step, 1)
    mu_lst = []
    grad_lst = []    
    grad = [grad_fn(mu)]
    mu = [mu_init]
    for i in range(max_iter):
        _, out = lax.scan(update_gradient, init=(mu[-1], grad[-1]), xs=step_grid)
        mu = out[0]
        grad = out[1]
        mu_lst.append(mu)
        grad_lst.append(grad)
        if jnp.linalg.norm(grad) < tol:
            _, out = lax.scan(update_newton, init=(mu[-1], grad[-1]), xs=step_grid)
            mu = out[0]
            grad = out[1]
            mu_lst.append(mu)
            grad_lst.append(grad)
            
    mu = jnp.stack(mu_lst)
    grad = jnp.stack(grad_lst)
    
    return mu, grad

#%% Joint Gradient-Newton Method

@jit
def JointGradientNewton(mu_rm:jnp.ndarray,
                       mu_euc:jnp.ndarray,
                       M:object,
                       grad_fn_rm:Callable[[jnp.ndarray], jnp.ndarray],
                       grad_fn_euc:Callable[[jnp.ndarray], jnp.ndarray],
                       ggrad_fn_rm:Callable[[jnp.ndarray], jnp.ndarray] = None,
                       ggrad_fn_euc:Callable[[jnp.ndarray], jnp.ndarray] = None,
                       step_size_rm:float = 0.1,
                       step_size_euc:float = 0.1,
                       iter_step:int=10,
                       tol:float=1e-1,
                       max_iter:int=100,
                       bnds_rm:Tuple[jnp.ndarray, jnp.ndarray] = (None, None),
                       bnds_euc:Tuple[jnp.ndarray, jnp.ndarray] = (None, None),
                       max_step:jnp.ndarray = None
                       )->Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    @jit
    def update_gradient(carry:Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], idx:int
                        )->Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                 Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        
        mu_rm, mu_euc, grad_rm, grad_euc = carry
        
        grad_rm = jnp.clip(grad_rm, min_step, max_step)
        grad_euc = jnp.clip(grad_euc, min_step, max_step)
        
        mu_rm = M.Exp(mu, -step_size_rm*grad_rm)
        mu_euc -= step_size_euc*grad_euc
        
        mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        mu_euc = jnp.clip(mu_euc, lb_euc, ub_euc)
    
        new_chart = M.centered_chart(mu_rm)
        mu_rm = M.update_coords(mu_rm,new_chart)
        
        grad_rm = grad_fn_rm(mu_rm)
        grad_euc = grad_fn_euc(mu_euc)
        out = (mu_rm, mu_euc, grad_rm, grad_euc)
    
        return out, out
    
    @jit
    def update_newton(carry:Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], idx:int
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
        mu_euc -= step_size_euc*step_grad_euc
        
        mu_rm[0] = jnp.clip(mu_rm[0], lb_rm, ub_rm)
        mu_euc = jnp.clip(mu_euc, lb_rm, ub_rm)
        
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
        
    step_grid = jnp.arange(0, iter_step, 1)
    mu_rm_lst = []
    mu_euc_lst = []
    grad_rm_lst = []
    grad_euc_lst = []    
    grad_rm = [grad_fn_rm(mu_rm)]
    grad_euc = [grad_fn_rm(mu_euc)]
    mu_rm = [mu_rm]
    mu_euc = [mu_rm]
    for i in range(max_iter):
        _, out = lax.scan(update_gradient, init=(mu_rm[-1], mu_euc[-1], 
                                                 grad_rm[-1], grad_euc[-1]), 
                          xs=step_grid)
        mu_rm = out[0]
        mu_euc = out[1]
        grad_rm = out[2]
        grad_euc = out[3]
        mu_rm_lst.append(mu_rm)
        mu_euc_lst.append(mu_euc)
        grad_rm_lst.append(grad_rm)
        grad_euc_lst.append(grad_euc)
        if jnp.linalg.norm(jnp.stack((grad_rm, grad_euc))) < tol:
            _, out = lax.scan(update_newton, init=(mu_rm[-1], mu_euc[-1], 
                                                     grad_rm[-1], grad_euc[-1]), 
                              xs=step_grid)
            mu_rm = out[0]
            mu_euc = out[1]
            grad_rm = out[2]
            grad_euc = out[3]
            mu_rm_lst.append(mu_rm)
            mu_euc_lst.append(mu_euc)
            grad_rm_lst.append(grad_rm)
            grad_euc_lst.append(grad_euc)
            
    mu_rm = jnp.stack(mu_rm_lst)
    mu_euc = jnp.stack(mu_euc_lst)
    grad_rm = jnp.stack(grad_rm_lst)
    grad_euc = jnp.stack(grad_euc_lst)
    
    return mu_rm, mu_euc, grad_rm, grad_euc

