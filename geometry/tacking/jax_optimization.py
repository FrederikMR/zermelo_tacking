#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M1:FinslerManifold,
                 M2:FinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.M1 = M1
        self.M2 = M2
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = self.init_default
            
        self.z0 = None
        self.zT = None
        self.idx = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def init_default(self, 
                     z0:Array, 
                     zT:Array,
                     idx:int,
                     T:int,
                     )->Array:
        
        z_mid = jnp.array([-jnp.cos(jnp.pi/4), jnp.sin(jnp.pi/4)])
        t1 = jnp.linspace(0.0,1.0,idx,endpoint=False, dtype=z0.dtype)[1:].reshape(-1,1)
        t2 = jnp.linspace(0.0,1.0,T-idx,endpoint=False, dtype=z0.dtype)[1:].reshape(-1,1)

        if len(t1)==0:
            curve1 = z0
        else:
            curve1 = z0+(z_mid-z0)*t1
        if len(t2)==0:
            curve2=zT
        else:
            curve2 = curve1[-1]+(zT-z_mid)*t2
        
        return jnp.vstack((curve1.reshape(-1,len(z0)), curve2.reshape(-1,len(z0))))
    
    def energy(self, 
               zt:Array, 
               )->Array:
        
        zt1 = zt[:self.idx]
        zt2 = zt[self.idx:]

        term1 = zt1[0]-self.z0
        val1 = self.M1.F(self.z0, term1)**2
        
        term2 = zt1[1:]-zt1[:-1]
        val2 = vmap(lambda x,v: self.M1.F(x,v)**2)(zt1[:-1], term2)
        
        term3 = zt2[0]-zt1[-1]
        val3 = self.M1.F(zt1[-1], term3)**2
        
        term4 = zt2[1:]-zt2[:-1]
        val4 = vmap(lambda x,v: self.M2.F(x,v)**2)(zt2[:-1], term4)
        
        term5 = self.zT-zt2[-1]
        val5 = self.M2.F(zt2[-1], term5)**2

        return val1+jnp.sum(val2)+val3+jnp.sum(val4)+val5
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        zt, opt_state = carry
        
        grad = self.Denergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 idx:int,
                 )->Array:
        print(idx)
        zt = self.init_fun(z0,zT,idx,self.T)
        
        self.idx = idx
        self.z0 = z0
        self.zT = zT

        opt_state = self.opt_init(zt)
        
        _, val = lax.scan(self.for_step,
                          init=(zt, opt_state),
                          xs = jnp.ones(self.max_iter),
                          )
        
        zt = val[0][-1]
    
        #zt = jnp.vstack((z0, zt, zT))
        
        return zt