#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.lorentz_finsler.manifolds import LorentzFinslerManifold

#%% JAX Optimization

class JAXOptimization(ABC):
    def __init__(self,
                 M:List[LorentzFinslerManifold],
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
    
    def time_update(self,
                    zt:Array,
                    )->Array:

        zt = zt.reshape(self.n_curves, -1, self.dim)

        t1 = self.time_step(0.0, zt[1][0], jnp.vstack((self.z0, zt[0])), self.M[0])
        ts = t1[-1]
        t1 = t1[:-1]
        if self.N_tacks>1:
            ztacks = zt[1:-1]
            times = []
            for i in range(1,len(ztacks)+1):
                t = self.time_step(ts, zt[i+1][0], zt[i], self.M[i])
                times.append(jnp.hstack(ts, t[:-1]))
                ts = t[-1]
            times = jnp.stack(times)

        tT = self.time_step(ts, self.zT, zt[-1], self.M[self.N_tacks])
        tT = jnp.hstack((ts, tT[:-1]))

        if self.N_tacks>1:
            t = jnp.hstack((t1, times, tT))
        else:
            t = jnp.hstack((t1, tT))
            
        return t
    
    def time_step(self,
                  t0:Array,
                  zT:Array,
                  z:Array,
                  M:LorentzFinslerManifold,
                  )->Array:
        
        def euler_step(t:Array,
                       step:Tuple[Array, Array],
                       )->Array:

            z, dz = step

            t += M.F(t,z,dz/self.dt)*self.dt
            
            return (t,)*2
        
        dz = z[1:]-z[:-1]
        dz = jnp.vstack((dz, zT-z[-1]))
        
        _, t = lax.scan(euler_step,
                        init=t0,
                        xs=(z,dz),
                        )
        
        return t
    
    def init_length(self,
                    t:Array,
                    zt:Array,
                    zT:Array,
                    M:LorentzFinslerManifold,
                    *args
                    )->Array:
        
        return M.length(t,jnp.vstack((self.z0,zt,zT)))
    
    def end_length(self, 
                   t:Array,
                   zt:Array,
                   M:LorentzFinslerManifold,
                   *args
                   )->Array:
        
        return M.length(t, jnp.vstack((zt,self.zT)))
    
    def mid_length(self, 
                   t:Array,
                   zt:Array,
                   zT:Array,
                   M:LorentzFinslerManifold,
                   *args
                   )->Array:
        
        return M.length(t, jnp.vstack((zt,zT)))
    
    def length(self, 
               t:Array,
               zt:Array, 
               *args
               )->Array:
        
        z0 = zt[0]
        zT = zt[-1]
        zt = zt[1:-1].reshape(self.n_curves, -1, self.dim)
        t = t.reshape(self.n_curves, -1)
        
        zt_first = zt[0]
        zT_first = zt[1][0]
        zt_end = zt[-1]
        M0 = self.M[0]
        MT = self.M[self.N_tacks]

        l1 = self.init_length(t[0], zt_first, zT_first, M0)
        lT = self.end_length(t[-1], zt_end, MT)

        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = zt[1:-1]
            l_tacks = jnp.sum(jnp.stack([self.mid_length(t[i], zt[i],zt[i+1][0],self.M[i]) for i in range(1,len(ztacks)+1)]))
        else:
            l_tacks = 0.0
            
        return l1+lT+l_tacks
    
    def init_energy(self,
                    t:Array,
                    zt:Array,
                    zT:Array,
                    M:LorentzFinslerManifold,
                    *args
                    )->Array:
        
        term1 = zt[0]-self.z0
        val1 = M.F(0.0, self.z0, term1)**2
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda t,x,v: M.F(t,x,v)**2)(t[:-1], zt[:-1], term2)
        
        term3 = zT-zt[-1]
        val3 = M.F(t[-1],zt[-1], term3)**2
        
        return val1+jnp.sum(val2)+val3
    
    def end_energy(self, 
                   t:Array,
                   zt:Array,
                   M:LorentzFinslerManifold,
                   *args
                   )->Array:

        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda t,x,v: M.F(t,x,v))(t[:-1],zt[:-1], term1)
        
        term2 = self.zT-zt[-1]
        val2 = M.F(t[-1],zt[-1], term2)**2
        
        return jnp.sum(val1)+val2
    
    def mid_energy(self, 
                   t:Array,
                   zt:Array,
                   zT:Array,
                   M:LorentzFinslerManifold,
                   *args
                   )->Array:
        
        term1 = zt[1:]-zt[:-1]
        val1 = vmap(lambda t,x,v: M.F(t,x,v))(t[:-1], zt[:-1], term1)
        
        term3 = zT-zt[-1]
        val3 = M.F(t[-1], zt[-1], term3)**2
        
        return jnp.sum(val1)+val3
    
    def energy(self, 
               t:Array,
               zt:Array, 
               *args
               )->Array:
        
        zt = zt.reshape(self.n_curves, -1, self.dim)
        t = t.reshape(self.n_curves, -1)
        
        zt_first = zt[0]
        zT_first = zt[1][0]
        zt_end = zt[-1]
        M0 = self.M[0]
        MT = self.M[self.N_tacks]

        e1 = self.init_energy(t[0], zt_first, zT_first, M0)
        eT = self.end_energy(t[-1], zt_end, MT)
        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = zt[1:-1]
            e_tacks = jnp.sum(jnp.stack([self.mid_energy(t[i], zt[i],zt[i+1][0],self.M[i]) for i in range(1,len(ztacks)+1)]))
        else:
            e_tacks = 0.0
            
        return e1+eT+e_tacks
    
    def Denergy(self,
                t:Array,
                zt:Array,
                )->Array:
        
        return grad(self.energy, argnums=1)(t, zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        t, zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        t, zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(t, zt)
        t = self.time_update(zt).reshape(self.n_curves, -1)
        
        return (t, zt, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        t, zt, opt_state = carry
        
        grad = self.Denergy(t, zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        t = self.time_update(zt).reshape(self.n_curves, -1)
        
        return ((t, zt, opt_state),)*2
    
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
        self.dt = 1.0/(self.T+(self.T-1)*self.N_tacks)
        
        tack_time = jnp.linspace(0.0,1.0,N_tacks+2, endpoint=True)
        ztack = z0+(zT-z0)*tack_time.reshape(-1,1)
        
        z_init = self.init_fun(ztack[0],ztack[1], self.T).reshape(1,self.T-1, self.dim)
        zt_paths = jnp.stack([self.init_tacks(ztack[i], ztack[i+1], self.T-1) for i in range(1,self.n_curves)])
        zt = jnp.vstack((z_init, zt_paths))
        
        self.z0 = z0
        self.zT = zT

        opt_state = self.opt_init(zt)
        t = self.time_update(zt).reshape(self.n_curves, -1)
        
        if step == "while":
            grad = self.Denergy(t, zt)
        
            t, zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                                 self.while_step,
                                                 init_val=(t, zt, grad, opt_state, 0)
                                                 )
            zt_paths = zt
            zt = jnp.vstack((z0, zt.reshape(-1,self.dim), zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(t, zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            t, zt = val[0], val[1]
            
            grad = vmap(self.Denergy)(t, zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return t, zt, idx