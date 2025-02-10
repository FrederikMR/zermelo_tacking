#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:53:14 2024

@author: fmry
"""

#%% Modules

from geometry.setup import *

from geometry.manifolds import LorentzFinslerManifold
from geometry.geodesic import GEORCE_H
from geometry.geodesic import GEORCE_HStep

#%% GEORCE Estimation of Tack Points and Geodesics

class SequentialOptimizationBFGS(ABC):
    def __init__(self,
                 M:List[LorentzFinslerManifold],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 sub_iter:int=100,
                 line_search_params:Dict={},
                 )->None:
        
        self.M = M
        self.T = T
        self.sub_iter = sub_iter
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Tacking Computation Object using ADAM and GEORCE Optimizers"
    
    def tack_curve(self,
                  z_tacks:Array,
                  )->Array:
        
        z_tacks = jnp.vstack((self.z0,
                              z_tacks.reshape(-1, len(self.z0)),
                              self.zT,
                              ))
        
        t0 = self.t0
        zt_curves = []
        t_curves = []
        for i in range(self.n_curves):
            self.Geodesic.M = self.M[i]
            (t,zt, *_) = self.Geodesic(t0, z_tacks[i], z_tacks[i+1])
            t0 = t[-1]
            t_curves.append(t)
            zt_curves.append(zt[:-1])
            
        return jnp.stack(t_curves).reshape(-1), jnp.vstack((jnp.stack(zt_curves).reshape(-1, self.dim), self.zT))
    
    def travel_time(self,
                    z_tacks:Array,
                    )->Array:
        
        z_tacks = jnp.vstack((self.z0,
                              z_tacks.reshape(-1, len(self.z0)),
                              self.zT,
                              ))
        
        travel_time = self.t0
        for i in range(self.n_curves):
            travel_time = self.StepGeodesic(self.M[i], travel_time, z_tacks[i], z_tacks[i+1])[0][-1]
            
        return travel_time**2
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 n_tacks:int=1,
                 )->Array:
        
        t0 = t0.astype("float64")
        z0 = z0.astype("float64")
        zT = zT.astype("float64")

        self.StepGeodesic = GEORCE_HStep(init_fun = self.init_fun,
                                         T=self.T,
                                         iters=self.sub_iter,
                                         line_search_params=self.line_search_params,
                                         )
        self.Geodesic = GEORCE_H(M=self.M[0],
                                 init_fun = self.init_fun,
                                 T=self.T,
                                 tol=self.tol,
                                 max_iter=self.max_iter,
                                 line_search_params=self.line_search_params,
                                 )
        
        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        self.n_tacks = n_tacks
        self.n_curves = n_tacks+1
        self.dim = len(z0)
        
        tack_times = jnp.linspace(0,1,n_tacks+1, endpoint=False)[1:]
        z_tacks = z0+jnp.einsum('i,t->ti', zT-z0, tack_times)
        
        res = jminimize(fun=self.travel_time,
                        x0=z_tacks.reshape(-1),
                        method="BFGS",
                        tol=self.tol,
                        options={'maxiter':self.max_iter},
                        )
        
        #t = res.fun
        z_tacks = res.x.reshape(-1, len(self.z0))
        t, zt = self.tack_curve(z_tacks)
        grad = res.jac
        idx = res.nit
        
        return t, zt, grad, idx
    
#%% GEORCE Estimation of Tack Points and Geodesics

class SequentialOptimizationADAM(ABC):
    def __init__(self,
                 M:List[LorentzFinslerManifold],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=0.01,
                 optimizer=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 sub_iter:int=100,
                 line_search_params:Dict={},
                 )->None:
        
        self.M = M
        self.T = T
        self.sub_iter = sub_iter
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_params = line_search_params
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Tacking Computation Object using ADAM and GEORCE Optimizers"
    
    def tack_curve(self,
                  z_tacks:Array,
                  )->Array:
        
        z_tacks = jnp.vstack((self.z0,
                              z_tacks.reshape(-1, len(self.z0)),
                              self.zT,
                              ))
        
        t0 = self.t0
        zt_curves = []
        t_curves = []
        for i in range(self.n_curves):
            self.Geodesic.M = self.M[i]
            (t,zt, *_) = self.Geodesic(t0, z_tacks[i], z_tacks[i+1])
            t0 = t[-1]
            t_curves.append(t)
            zt_curves.append(zt[:-1])
            
        return jnp.stack(t_curves).reshape(-1), jnp.vstack((jnp.stack(zt_curves).reshape(-1, self.dim), self.zT))
    
    def travel_time(self,
                    z_tacks:Array,
                    )->Array:
        
        z_tacks = jnp.vstack((self.z0,
                              z_tacks.reshape(-1, len(self.z0)),
                              self.zT,
                              ))
        
        travel_time = self.t0
        for i in range(self.n_curves):
            travel_time = self.StepGeodesic(self.M[i], travel_time, z_tacks[i], z_tacks[i+1])[0][-1]
            
        return travel_time**2
    
    def Dtime(self,
              z_tacks:Array,
              )->Array:
        
        return grad(self.travel_time, argnums=0)(z_tacks)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        z_tacks, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, int],
                   )->Array:
        
        z_tacks, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        z_tacks = self.get_params(opt_state)
        grad = self.Dtime(z_tacks)
        
        return (z_tacks, grad, opt_state, idx+1)
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 n_tacks:int=1,
                 )->Array:
        
        t0 = t0.astype("float64")
        z0 = z0.astype("float64")
        zT = zT.astype("float64")
        
        self.StepGeodesic = GEORCE_HStep(init_fun = self.init_fun,
                                         T=self.T,
                                         iters=self.sub_iter,
                                         line_search_params=self.line_search_params,
                                         )
        self.Geodesic = GEORCE_H(M=self.M[0],
                                 init_fun = self.init_fun,
                                 T=self.T,
                                 tol=self.tol,
                                 max_iter=self.max_iter,
                                 line_search_params=self.line_search_params,
                                 )
        
        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        self.n_tacks = n_tacks
        self.n_curves = n_tacks+1
        self.dim = len(z0)
        
        tack_times = jnp.linspace(0,1,n_tacks+1, endpoint=False)[1:]
        z_tacks = z0+jnp.einsum('i,t->ti', zT-z0, tack_times)
        
        opt_state = self.opt_init(z_tacks)
        grad = self.Dtime(z_tacks)
    
        z_tacks, grad, opt_state, idx = lax.while_loop(self.cond_fun, 
                                                       self.while_step,
                                                       init_val=(z_tacks, grad, opt_state, 0)
                                                       )

        t, zt = self.tack_curve(z_tacks)

        return t, zt, grad, idx

