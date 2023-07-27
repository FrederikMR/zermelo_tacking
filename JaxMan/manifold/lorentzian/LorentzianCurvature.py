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

def LorentzianCurvature(M: object) -> None:
    
    @jit
    def CurvatureOperator(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        G = M.G(t, x)
        
        chris = M.Chris(t, x)        
        Dchris = M.DChris(t, x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            - jnp.einsum('mikj->mijk', Dchris) \
            + jnp.einsum('sjk,mis->mijk', chris, chris) \
            - jnp.einsum('sik,mjs->mijk', chris, chris)
        
        return R
    
    @jit
    def CurvatureTensor(t:float, x: jnp.ndarray)  -> jnp.ndarray:
        
        G = M.G(t, x)
        
        chris = M.Chris(t, x)        
        Dchris = M.DChris(t, x)
        CO = jnp.einsum('mjki->mijk', Dchris) \
            - jnp.einsum('mikj->mijk', Dchris) \
            + jnp.einsum('sjk,mis->mijk', chris, chris) \
            - jnp.einsum('sik,mjs->mijk', chris, chris)
        
        return jnp.einsum('sijk,sm->ijkm', CO, G)
    
    @jit
    def SectionalCurvature(t:float, x: jnp.ndarray, e1:jnp.ndarray, e2:jnp.ndarray) -> jnp.ndarray:
        
        G = M.G(t,x)
        CO = M.CurvatureOperator(t,x)
        
        CT = jnp.einsum('sijk,sm->ijkm', CO, G)[0,1,1,0]
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    @jit
    def RicciCurvature(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('kijk->ij', M.CurvatureOperator(t, x))
    
    @jit
    def ScalarCurvature(t:float, x:jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('ij,ij->', M.Ginv(t, x), M.RicciCurvature(t, x))
    
    @jit
    def TracelessRicci(t:float, x:jnp.ndarray)->jnp.ndarray:
        
        G = M.G(t, x)
        R = M.RicciCurvature(t, x)
        S = M.ScalarCurvature(t, x)
        
        return R-S*G/M.dim
    
    @jit
    def EinsteinTensor(t:float, x: jnp.ndarray) -> jnp.ndarray:
        
        R = M.RicciCurvature(t, x)
        S = M.ScalarCurvature(t, x)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    