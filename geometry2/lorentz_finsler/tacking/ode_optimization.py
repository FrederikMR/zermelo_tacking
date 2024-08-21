#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.finsler.geodesics import GEORCE
from geometry.finsler.manifolds import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class ODEOptimization(ABC):
    def __init__(self,
                 M:List[FinslerManifold],
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        self.init_fun = lambda z0, zT, T, end_point=False: (zT-z0)*jnp.linspace(0.0,
                                                                                1.0,
                                                                                T,
                                                                                endpoint=False,
                                                                                dtype=z0.dtype)[1:].reshape(-1,1)+z0
        self.init_tacks = lambda z0, zT, T, end_point=False: (zT-z0)*jnp.linspace(0.0,
                                                                                  1.0,
                                                                                  T,
                                                                                  endpoint=False,
                                                                                  dtype=z0.dtype).reshape(-1,1)+z0
            
        self.z0 = None
        self.zT = None
        self.dt_steps = jnp.ones(self.T, dtype=jnp.float32)/self.T
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def euler_step(self, 
                   carry:Tuple[Array, Array, Array],
                   dt:float,
                   )->Tuple[Array, Array, Array]:
        
        t, z, v = carry
        
        t += dt
        df = self.geodesic_equation(z,v)*dt
        
        z += df[0]
        v += df[1]
        
        return ((t, z, v),)*2
    
    def geodesic(self,
                M:FinslerManifold,
                x:Array,
                y:Array,
                v:Array,
                )->Array:
        
        self.geodesic_equation = M.geodesic_equation
        _, val = lax.scan(self.euler_step,
                          init=(0.0, x, v),
                          xs=self.dt_steps)
        t, gamma, gammav = val
        
        return jnp.vstack((x, gamma)), gammav
    
    def obj_fun(self,
                x:Array
                ):
        
        x = x.reshape(-1,self.dim)
        z_tacks = x[:self.n_tacks].reshape(-1,self.dim)
        v = x[self.n_tacks:].reshape(-1,self.dim)
        
        gamma1, gamma1v = self.geodesic(self.M[0], self.z0, z_tacks[0], v[0])
        error1 = jnp.sum(jnp.square(gamma1[-1]-z_tacks[0]))
        length1 = self.M[0].length(gamma1)
        
        lengtht = 0.0
        errort = 0.0
        for i in range(self.n_tacks-1):
            gammat, gammatv = self.geodesic(self.M[i+1], z_tacks[i], z_tacks[i+1], v[i+1])
            lengtht += self.M[i+1].length(gammat)
            errort += jnp.sum(jnp.square(gammat[-1]-z_tacks[i+1]))
            
        gammaT, gammaTv = self.geodesic(self.M[self.n_tacks], z_tacks[-1], self.zT, v[-1])
        errorT = jnp.sum(jnp.square(gammaT[-1]-self.zT))
        lengthT = self.M[self.n_tacks].length(gammaT)

        return error1+errort+errorT+length1+lengtht+lengthT
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 n_tacks:int=None,
                 )->Array:
        
        if n_tacks is None:
            self.n_tacks = 1
        else:
            self.n_tacks = n_tacks
        self.n_curves = self.n_tacks+1
        
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)
        tack_time = jnp.linspace(0.0,1.0,self.n_tacks+1, endpoint=True)[1:]
        ztack_init = z0+(zT-z0)*tack_time.reshape(-1,1)
        v_init = jnp.ones((self.n_curves, self.dim))
        
        x0 = jnp.hstack((ztack_init.reshape(-1), v_init.reshape(-1)))
        
        opt = jminimize(self.obj_fun, x0=x0, method="BFGS", tol=1e-16, options={'maxiter':10000})
        
        print(opt.jac)
        print(opt.fun)
        print(opt.nit)
        
        x = opt.x
        
        x = x.reshape(-1,self.dim)
        z_tacks = x[:self.n_tacks]
        v = x[self.n_tacks:]
        
        gamma1, gamma1v = self.geodesic(self.M[0], self.z0, z_tacks[0], v[0])
        gammat = []
        for i in range(self.n_tacks-1):
            gammat.append(self.geodesic(self.M[i+1], z_tacks[i], z_tacks[i+1], v[i+1])[0])
            
        gammaT, gammaTv = self.geodesic(self.M[self.n_tacks], z_tacks[-1], self.zT, v[-1])
        
        if self.n_tacks > 1:
            return jnp.concatenate((gamma1.reshape(1,-1,self.dim), jnp.stack(gammat), gammaT.reshape(1,-1,self.dim)),
                                   axis=0), z_tacks, v
        else:
            return jnp.concatenate((gamma1.reshape(1,-1,self.dim), gammaT.reshape(1,-1,self.dim)), 
                                   axis=0), z_tacks, v