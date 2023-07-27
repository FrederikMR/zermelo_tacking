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

def RiemannianMetric(M: object) -> None:
    
    @jit
    def Measure(x: jnp.ndarray) -> jnp.ndarray:
        
        sgn, logdet = jnp.linalg.slogdet(M.G(x))
        
        return sgn*jnp.exp(logdet)
    
    @jit
    def DG(x:jnp.ndarray) -> jnp.ndarray:
        
        return jacfwd(M.G)(x)
    
    @jit
    def Ginv(x:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.linalg.inv(M.G(x))
    
    @jit
    def dot(x:jnp.ndarray, v: jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.tensordot(jnp.tensordot(M.G(x),w,(1,0)),v,(0,0))
    
    @jit
    def norm2(x: jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        return M.dot(x,v,v)
    
    @jit
    def Chris(x: jnp.ndarray) -> jnp.ndarray:
        
        DG = M.DG(x)
        Ginv = M.Ginv(x)
        
        return 0.5*(jnp.einsum('mk,kij->mij', Ginv, DG) \
                    + jnp.einsum('mk,kji->mij', Ginv, DG) \
                    - jnp.einsum('mk,ijk->mij', Ginv, DG))
            
    @jit
    def DChris(x: jnp.ndarray) -> jnp.ndarray:
        
        return jacfwd(M.Chris)(x)
    
    @partial(jit, static_argnames=['f'])
    def Grad(x: jnp.ndarray, f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return M.Ginv(x).dot(grad(f)(x))
    
    @partial(jit, static_argnames=['V'])
    def Div(x: jnp.ndarray, V: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return jnp.trace(jacfwd(V)(x))+jnp.dot(V(x),grad(M.Measure)(x))
    
    @partial(jit, static_argnames=['f'])
    def LaplaceBeltrami(x: jnp.ndarray, f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        
        return M.Div(x, lambda x: M.Grad(x, f))
            
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
    
                
    