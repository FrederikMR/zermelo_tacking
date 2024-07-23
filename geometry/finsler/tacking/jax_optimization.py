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

class JAXOptimization(ABC):
    def __init__(self,
                 M:List[FinslerManifold],
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 )->None:
        
        self.Geodesic = lambda z0,zT,M: jnp.linspace(0,1,self.T+1, endpoint=True).reshape(-1,1)*(zT-z0)+z0
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
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def init_length(self,
                    zt:Array,
                    zT:Array,
                    M:FinslerManifold,
                    *args
                    )->Array:
        
        return M.length(jnp.vstack((self.z0,zt,zT)))
    
    def end_length(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        return M.length(jnp.vstack((zt,self.zT)))
    
    def length_fun(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda x,v: M.F(x,v))(zt[:-1], term1)
        
        return jnp.sum(val1)

    def length(self, 
               ztacks:Array, 
               *args
               )->Array:
        
        e1 = self.length_fun(self.Geodesic(self.z0,ztacks[0],self.M[0]), self.M[0])
        eT = self.length_fun(self.Geodesic(ztacks[-1],self.zT,self.M[self.N_tacks]), self.M[self.N_tacks])
        
        if self.N_tacks>1:
            e_tacks = jnp.sum(jnp.stack([self.length_fun(self.Geodesic(ztacks[i],ztacks[i+1],self.M[i]), self.M[i]) for i in range(len(ztacks)-1)]))
        else:
            e_tacks = 0.0
            
        return e1+eT+e_tacks
    
    def init_energy(self,
                    zt:Array,
                    zT:Array,
                    M:FinslerManifold,
                    *args
                    )->Array:
        
        term1 = zt[0]-self.z0
        val1 = M.F(self.z0, term1)**2
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term2)
        
        term3 = zT-zt[-1]
        val3 = M.F(zt[-1], term3)**2
        
        return val1+jnp.sum(val2)+val3
    
    def end_energy(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:

        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term1)
        
        term2 = self.zT-zt[-1]
        val2 = M.F(zt[-1], term2)**2
        
        return jnp.sum(val1)+val2
    
    def energy_fun(self,
                  zt:Array,
                  M:FinslerManifold,
                  *args
                  )->Array:
        
        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term1)
        
        return jnp.sum(val1)
    
    def energy(self, 
               ztacks:Array, 
               *args
               )->Array:
        
        e1 = self.energy_fun(self.Geodesic(self.z0,ztacks[0],self.M[0]), self.M[0])
        eT = self.energy_fun(self.Geodesic(ztacks[-1],self.zT,self.M[self.N_tacks]), self.M[self.N_tacks])
        
        if self.N_tacks>1:
            e_tacks = jnp.sum(jnp.stack([self.energy_fun(self.Geodesic(ztacks[i],ztacks[i+1],self.M[i]), self.M[i]) for i in range(len(ztacks)-1)]))
        else:
            e_tacks = 0.0
            
        return e1+eT+e_tacks
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
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
                 N_tacks:int=None,
                 step:str="while",
                 )->Array:
                
        if N_tacks is None:
            N_tacks = len(self.M)-1
            
        self.n_curves = N_tacks+1
        self.N_tacks = N_tacks
        self.dim = len(z0)
        
        tack_time = jnp.linspace(0.0,1.0,N_tacks+2, endpoint=True)
        ztack = z0+(zT-z0)*tack_time.reshape(-1,1)
        ztack = ztack[1:-1].reshape(-1,self.dim)
        
        self.z0 = z0
        self.zT = zT

        opt_state = self.opt_init(ztack)
        
        if step == "while":
            grad = self.Denergy(ztack)
        
            ztacks, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(ztack, grad, opt_state, 0)
                                              )
            print(jnp.linalg.norm(grad.reshape(-1)))
            
            zt_0 = self.Geodesic(self.z0,ztacks[0],self.M[0])
            zt_T = self.Geodesic(ztacks[-1],self.zT,self.M[0])[1:]
            
            if self.N_tacks>1:
                zt_t = jnp.stack([self.Geodesic(ztacks[i],ztacks[i+1],self.M[i])[1:] for i in range(len(ztacks)-1)])
                zt = jnp.vstack((zt_0, zt_t.reshape(-1,self.dim), zt_T))
            else:
                zt = jnp.vstack((zt_0, zt_T))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(ztack, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt
    
#%%

class JAXOptimization2(ABC):
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
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def init_length(self,
                    zt:Array,
                    zT:Array,
                    M:FinslerManifold,
                    *args
                    )->Array:
        
        return M.length(jnp.vstack((self.z0,zt,zT)))
    
    def end_length(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        return M.length(jnp.vstack((zt,self.zT)))
    
    def mid_length(self, 
                   zt:Array,
                   zT:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        return M.length(jnp.vstack((zt,zT)))
    
    def length(self, 
               zt:Array, 
               *args
               )->Array:
        
        z0 = zt[0]
        zT = zt[-1]
        zt = zt[1:-1].reshape(self.n_curves, -1, self.dim)
        
        zt_first = zt[0]
        zT_first = zt[1][0]
        zt_end = zt[-1]
        M0 = self.M[0]
        MT = self.M[self.N_tacks]

        l1 = self.init_length(zt_first, zT_first, M0)
        lT = self.end_length(zt_end, MT)

        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = zt[1:-1]
            l_tacks = jnp.sum(jnp.stack([self.mid_length(zt[i],zt[i+1][0],self.M[i]) for i in range(1,len(ztacks)+1)]))
        else:
            l_tacks = 0.0
            
        return l1+lT+l_tacks
    
    def init_energy(self,
                    zt:Array,
                    zT:Array,
                    M:FinslerManifold,
                    *args
                    )->Array:
        
        term1 = zt[0]-self.z0
        val1 = M.F(self.z0, term1)**2
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term2)
        
        term3 = zT-zt[-1]
        val3 = M.F(zt[-1], term3)**2
        
        return val1+jnp.sum(val2)+val3
    
    def end_energy(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:

        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term1)
        
        term2 = self.zT-zt[-1]
        val2 = M.F(zt[-1], term2)**2
        
        return jnp.sum(val1)+val2
    
    def mid_energy(self, 
                   zt:Array,
                   zT:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda x,v: M.F(x,v)**2)(zt[:-1], term1)
        
        term3 = zT-zt[-1]
        val3 = M.F(zt[-1], term3)**2
        
        return jnp.sum(val1)+val3
    
    def energy(self, 
               zt:Array, 
               *args
               )->Array:
        
        zt = zt.reshape(self.n_curves, -1, self.dim)
        
        zt_first = zt[0]
        zT_first = zt[1][0]
        zt_end = zt[-1]
        M0 = self.M[0]
        MT = self.M[self.N_tacks]

        e1 = self.init_energy(zt_first, zT_first, M0)
        eT = self.end_energy(zt_end, MT)
        #e2 = self.mid_energy(zt[1], zt[2][0], self.M[1])
        
        #return e1+e2+eT
        
        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = zt[1:-1]
            e_tacks = jnp.sum(jnp.stack([self.mid_energy(zt[i],zt[i+1][0],self.M[i]) for i in range(1,len(ztacks)+1)]))
        else:
            e_tacks = 0.0
            
        return e1+eT+e_tacks
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
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
                 N_tacks:int=None,
                 step:str="while",
                 )->Array:
                
        if N_tacks is None:
            N_tacks = len(self.M)-1
            
        self.n_curves = N_tacks+1
        self.N_tacks = N_tacks
        self.dim = len(z0)
        
        tack_time = jnp.linspace(0.0,1.0,N_tacks+2, endpoint=True)
        ztack = z0+(zT-z0)*tack_time.reshape(-1,1)
        
        z_init = self.init_fun(ztack[0],ztack[1], self.T).reshape(1,self.T-1, self.dim)
        zt_paths = jnp.stack([self.init_tacks(ztack[i], ztack[i+1], self.T-1) for i in range(1,self.n_curves)])
        zt = jnp.vstack((z_init, zt_paths))
        
        self.z0 = z0
        self.zT = zT

        opt_state = self.opt_init(zt)
        
        if step == "while":
            grad = self.Denergy(zt)
        
            zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(zt, grad, opt_state, 0)
                                              )
            print(jnp.linalg.norm(grad.reshape(-1)))
            zt_paths = zt
            zt = jnp.vstack((z0, zt.reshape(-1,self.dim), zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt