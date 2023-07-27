#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:30:20 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JaxMan.initialize import *

#%% RiemannianMetric

def LorentzianMetric(M: object) -> None:
    
    @jit
    def Measure(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        sgn, logdet = jnp.linalg.slogdet(M.G(t,x))
        
        return sgn*jnp.exp(logdet)
    
    @jit
    def DG(t:float, x:jnp.ndarray) -> jnp.ndarray:
        
        return jacfwd(lambda y: M.G(y[0], y[1:]))(jnp.hstack((t,x)))
    
    @jit
    def Ginv(t:float, x:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.linalg.inv(M.G(t, x))
    
    @jit
    def dot(t:float, x:jnp.ndarray, v: jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.tensordot(jnp.tensordot(M.G(t, x),w,(1,0)),v,(0,0))
    
    @jit
    def norm2(t:float, x: jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        return M.dot(t, x,v,v)
    
    @jit
    def Chris(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        DG = M.DG(t,x)
        Ginv = M.Ginv(t,x)
        
        return 0.5*(jnp.einsum('mk,kij->mij', Ginv, DG) \
                    + jnp.einsum('mk,kji->mij', Ginv, DG) \
                    - jnp.einsum('mk,ijk->mij', Ginv, DG))
            
    @jit
    def DChris(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        return jacfwd(lambda y: M.Chris(y[0], y[1:]))(jnp.hstack((t,x)))
    
    @partial(jit, static_argnames=['f'])
    def Grad(t:float, x: jnp.ndarray, f: Callable[[float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return M.Ginv(t, x).dot(grad(f)(t, x))
    
    @partial(jit, static_argnames=['V'])
    def Div(t:float, x: jnp.ndarray, V: Callable[[float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return jnp.trace(jacfwd(V)(t,x))+jnp.dot(V(t,x),grad(M.Measure)(t,x))
    
    @partial(jit, static_argnames=['f'])
    def LaplaceBeltrami(t:float, x: jnp.ndarray, f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return M.Div(t, x, lambda x: M.Grad(t, x, f))
            
    M.Measure = Measure
    M.DG = DG
    M.Ginv = Ginv
    M.Chris = Chris
    M.DChris = DChris
    
    M.dot = dot
    M.norm2 = norm2
    
    M.Grad = Grad
    M.Div = Div
    M.LaplaceBeltrami = LaplaceBeltrami
    
    return
