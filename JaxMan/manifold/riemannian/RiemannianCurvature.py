#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:28:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JaxMan.initialize import *

#%% Curvature Code

def RiemannianCurvature(M: object) -> None:
    
    @jit
    def CurvatureOperator(x: jnp.ndarray) -> jnp.ndarray:
        
        G = M.G(x)
        
        chris = M.Chris(x)        
        Dchris = M.DChris(x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            - jnp.einsum('mikj->mijk', Dchris) \
            + jnp.einsum('sjk,mis->mijk', chris, chris) \
            - jnp.einsum('sik,mjs->mijk', chris, chris)
        
        return R
    
    @jit
    def CurvatureTensor(x: jnp.ndarray)  -> jnp.ndarray:
        
        G = M.G(x)
        
        chris = M.Chris(x)        
        Dchris = M.DChris(x)
        CO = jnp.einsum('mjki->mijk', Dchris) \
            - jnp.einsum('mikj->mijk', Dchris) \
            + jnp.einsum('sjk,mis->mijk', chris, chris) \
            - jnp.einsum('sik,mjs->mijk', chris, chris)
        
        return jnp.einsum('sijk,sm->ijkm', CO, G)
    
    @jit
    def SectionalCurvature(x: jnp.ndarray, e1:jnp.ndarray, e2:jnp.ndarray) -> jnp.ndarray:
        
        G = M.G(x)
        CO = M.CurvatureOperator(x)
        
        CT = jnp.einsum('sijk,sm->ijkm', CO, G)[0,1,1,0]
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    @jit
    def RicciCurvature(x: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('kijk->ij', M.CurvatureOperator(x))
    
    @jit
    def ScalarCurvature(x:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('ij,ij->', M.Ginv(x), M.RicciCurvature(x))
    
    @jit
    def TracelessRicci(x:jnp.ndarray)->jnp.ndarray:
        
        G = M.G(x)
        R = M.RicciCurvature(x)
        S = M.ScalarCurvature(x)
        
        return R-S*G/M.dim
    
    @jit
    def EinsteinTensor(x: jnp.ndarray) -> jnp.ndarray:
        
        R = M.RicciCurvature(x)
        S = M.ScalarCurvature(x)
        G = M.G(x)
        
        return R-0.5*S*G
        
    M.CurvatureOperator = CurvatureOperator
    M.CurvatureTensor = CurvatureTensor
    M.SectionalCurvature = SectionalCurvature
    M.RicciCurvature = RicciCurvature
    M.ScalarCurvature = ScalarCurvature
    M.TracelessRicci = TracelessRicci
    M.EinsteinTensor = EinsteinTensor
    
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    