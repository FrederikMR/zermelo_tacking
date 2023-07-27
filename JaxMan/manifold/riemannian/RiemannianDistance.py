#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:36:50 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JaxMan.initialize import *

#%% Riemannian Distances

def RiemannianDistance(M: object) -> None:
    
    @jit
    def EulerLagrange(x:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        chris = M.Chris(x)
        
        return jnp.concatenate((v, -jnp.einsum('i,j,kij->k', v, v, chris)))
    
    @jit
    def EnergyFunctional(gamma: jnp.ndarray, dgamma: jnp.ndarray) -> jnp.ndarray:
        
        norm2 = vmap(M.norm2)(gamma, dgamma)
        
        return M.IntegrationMethod(norm2, M.dt)
    
    @jit
    def LengthFunctional(gamma: jnp.ndarray, dgamma: jnp.ndarray) -> jnp.ndarray:
        
        norm2 = vmap(M.norm2)(gamma,dgamma)
        
        return M.IntegrationMethod(norm2, M.dt)
    
    @jit
    def Distance(x:jnp.ndarray, y:jnp.ndarray) -> jnp.ndarray:
        
        grid, gamma, dgamma = M.BVPGeodesic(x,y)[1:]
        
        norm = jnp.sqrt(vmap(M.norm2)(gamma, dgamma))
        
        return M.IntegrationMethod(norm, M.dt)
    
    @partial(jit, static_argnames=['T'])
    def IVPGeodesic(x:jnp.ndarray, v:jnp.ndarray, T:float=1.0) -> jnp.ndarray:
        
        y0 = jnp.concatenate((x,v), axis=0)
        grid = jnp.arange(0,T+M.dt,M.dt)
        
        y = M.ODEMethod(lambda _, x: M.EulerLagrange(x[:M.dim], x[M.dim:]), y0,
                        grid)
        
        gamma = y[:, :M.dim]
        dgamma = y[:,M.dim:]
        
        return grid, gamma, dgamma
    
    @partial(jit, static_argnames=['T'])
    def BVPODE(x:jnp.ndarray, y:jnp.ndarray, 
               T:float = 1.0) -> jnp.ndarray:
        
        @jit
        def error_fun(v):
            
            y0 = jnp.concatenate((x,v), axis=0)
            
            y1 = M.ODEMethod(f_fun, y0, grid)
            
            gamma = y1[:, :M.dim]
            
            return jnp.sum((gamma[-1]-y)**2)
        
        f_fun = jit(lambda _, x: M.EulerLagrange(x[:M.dim], x[M.dim:]))
        grid = jnp.arange(0,T+M.dt,M.dt)
        v0_init = y-x
        opt = minimize(error_fun,
                            v0_init,
                            method=M.optimizer,
                            tol=M.tol,
                            options={'maxiter': M.maxiter}
                            )
        
        v0 = opt.x
        #if not opt.success:
        #    print("WARNING - GEODESICS NOT SUCCESFULLY CONVERGED WITH STATUS: {}".format(opt.status))
        
        grid, gamma, dgamma = M.IVPGeodesic(x,v0,T)
        
        return grid, gamma, dgamma
    
    @partial(jit, static_argnames=['T'])
    def BVPEnergy(x:jnp.ndarray, y:jnp.ndarray, 
                  T:float = 1.0) -> jnp.ndarray:
        
        @jit
        def error_fun(gamma):
            
            gamma = jnp.concatenate((x, gamma.reshape(Nt, -1), y), axis=0)
            g1 = gamma[:-1]
            g2 = gamma[1:]
            
            dgamma = (g2-g1)/M.dt
            norm2 = vmap(lambda x,v: M.norm2(x,v))(g1, dgamma)
            
            return M.IntegrationMethod(norm2, M.dt)
        
        grid = jnp.arange(0,T+M.dt,M.dt)
        Nt = len(grid)-2
        gamma_init = (x+(y-x)*grid.reshape(-1,1))[1:-1]
            
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        
        opt = minimize(error_fun,
                       gamma_init.reshape(-1),
                       method=M.optimizer,
                       tol=M.tol,
                       options={'maxiter': M.maxiter}
                       )
        
        gamma = (opt.x).reshape(Nt,-1)
        gamma = jnp.concatenate((x, gamma, y), axis=0)
        dgamma = (gamma[1:]-gamma[:-1])/M.dt
        #if not opt.success:
        #    print("WARNING - GEODESICS NOT SUCCESFULLY CONVERGED WITH STATUS: {}".format(opt.status))
        
        return grid, gamma, dgamma
    
    @jit
    def Exp(x:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        return M.IVPGeodesic(x,v, 1.0)[1][-1]
    
    @jit
    def Log(x:jnp.ndarray, y:jnp.ndarray, v0_init:jnp.ndarray=jnp.ones(M.dim)) -> jnp.ndarray:
        
        return M.BVPGeodesic(x,y,v0_init)[1][0]
    
    @jit
    def ParallelTransport(x:jnp.ndarray, v:jnp.ndarray) -> jnp.ndarray:
        
        return
    
    M.EulerLagrange = EulerLagrange
    M.EnergyFunctional = EnergyFunctional
    M.Distance = Distance
    
    M.IVPGeodesic = IVPGeodesic
    
    if M.GeodesicMethod == "ODE":
        M.BVPGeodesic = BVPODE
    elif M.GeodesicMethod == "Energy":
        M.BVPGeodesic = BVPEnergy
    else:
        raise ValueError("Unsupported geodesic method")
    
    
    M.Exp = Exp
    M.Log = Log
    M.ParallelTransport = ParallelTransport
    
    return