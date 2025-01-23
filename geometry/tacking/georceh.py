#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:53:14 2024

@author: fmry
"""

#%% Modules

from geometry.setup import *

from geometry.manifolds import LorentzFinslerManifold
from geometry.line_search import Bisection, Backtracking

#%% GEORCE Estimation of Tack Points and Geodesics

class GEORCE_HTacking(ABC):
    def __init__(self,
                 M:List[LorentzFinslerManifold],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:
        
        self.M = M
        
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype).reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Tacking Computation Object using Control Problem"
    
    def init_curve(self,
                   )->Array:
        
        if self.n_tacks < 1.:
            return self.init_fun(self.z0,self.zT,self.T)[1:].reshape(self.T-1, -1)
        elif self.n_tacks < 2.:
            tack_times = jnp.linspace(0.0, 1.0, self.n_curves, endpoint=False)[1:]
            tack_points = (self.zT-self.z0)*tack_times.reshape(-1,1)
            zt1 = self.init_fun(self.z0, tack_points[0], self.T)[1:]
            ztT = self.init_fun(tack_points[-1], self.zT, self.T-1)
            
            return jnp.vstack((zt1, ztT))
        else:
            tack_times = jnp.linspace(0.0, 1.0, self.n_curves, endpoint=False)[1:]
            tack_points = self.z0+(self.zT-self.z0)*tack_times.reshape(-1,1)
            
            zt1 = self.init_fun(self.z0, tack_points[0], self.T)[1:]
            zt = jnp.stack([self.init_fun(tack_points[i], tack_points[i+1], self.T-1) for i in range(self.n_tacks-1)])
            ztT = self.init_fun(tack_points[-1], self.zT, self.T-1)
            
            return jnp.vstack((zt1, zt.reshape(-1,self.dim), ztT))
    
    def energy(self, 
               zs:Array, 
               *args
               )->Array:
        
        zs = zs.reshape(self.n_curves, -1, self.dim)
        
        if self.n_tacks < 1.:
            _, energy = self.path_energy(self.t0,
                                         self.z0,
                                         zs[0], 
                                         self.zT, 
                                         self.M[0],
                                         *args)
            
            return energy
        
        elif self.n_tacks < 2.:
            t0, e1 = self.path_energy(self.t0,
                                      self.z0,
                                      zs[0],
                                      zs[1][0],
                                      self.M[0],
                                      *args,
                                      )
            
            _, eT = self.path_energy(t0,
                                     zs[-1][0],
                                     zs[-1][1:],
                                     self.zT,
                                     self.M[self.n_tacks],
                                     *args,
                                     )
            
            return e1+eT
            
        else:
            t0, e1 = self.path_energy(self.t0,
                                      self.z0,
                                      zs[0],
                                      zs[1][0],
                                      self.M[0],
                                      *args,
                                      )
            energy = []
            for i in range(1,self.n_tacks):
                t0, e = self.path_energy(t0,
                                         zs[i][0],
                                         zs[i][1:],
                                         zs[i+1][0],
                                         self.M[i],
                                         *args,
                                         )
                energy.append(e)
                
            _, eT = self.path_energy(t0,
                                     zs[-1][0],
                                     zs[-1][1:],
                                     self.zT,
                                     self.M[self.n_tacks],
                                     *args,
                                     )
            
            return e1+jnp.sum(jnp.stack(energy))+eT
    
    def path_energy(self,
                    t0:Array,
                    z0:Array,
                    zs:Array,
                    zT:Array,
                    M:LorentzFinslerManifold,
                    *args,
                    )->Array:

        us = jnp.vstack((zs[0]-z0,
                         zs[1:]-zs[:-1],
                         zT-zs[-1]
                         ))
        ts = self.update_time_path(t0, z0, zs, zT, M)
        
        val1 = M.F(t0, z0, us[0])**2
        val2 = vmap(lambda t,x,v: M.F(t,x,v)**2)(ts[:-1], zs, us[1:])

        return ts[-1], val1+jnp.sum(val2)
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()

        return grad(self.energy)(zt)
    
    def Lt(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        ts = ts.reshape(self.n_curves, -1)
        zs = zs.reshape(self.n_curves, -1, self.dim)
        us = us.reshape(self.n_curves, -1, self.dim)
        
        Lt = [vmap(grad(self.M[i].F, argnums=0))(ts[i], zs[i], us[i]) for i in range(self.n_curves)]
        
        return jnp.stack(Lt).reshape(-1)
    
    def Lz(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        ts = ts.reshape(self.n_curves, -1)
        zs = zs.reshape(self.n_curves, -1, self.dim)
        us = us.reshape(self.n_curves, -1, self.dim)
        
        Lz = [vmap(jacfwd(self.M[i].F, argnums=1))(ts[i], zs[i], us[i]) for i in range(self.n_curves)]
        
        return jnp.stack(Lz).reshape(-1,self.dim)
    
    def Lu(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        ts = ts.reshape(self.n_curves, -1)
        zs = zs.reshape(self.n_curves, -1, self.dim)
        us = us.reshape(self.n_curves, -1, self.dim)
        
        Lu = [vmap(jacfwd(self.M[i].F, argnums=2))(ts[i], zs[i], us[i]) for i in range(self.n_curves)]
        
        return jnp.stack(Lu).reshape(-1,self.dim)
    
    def inner_product(self,
                      ts:Array,
                      zs:Array,
                      us:Array,
                      M:LorentzFinslerManifold,
                      )->Array:
        
        Gs = vmap(M.G)(ts,zs,us)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', us, Gs, us))
    
    def inner_product_h(self,
                        ts:Array,
                        zs:Array,
                        u0:Array,
                        us:Array,
                        M:LorentzFinslerManifold,
                        )->Array:
        
        Gs = vmap(M.G)(ts,zs,us)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', u0, Gs, u0))
    
    def rs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        zs = zs.reshape(self.n_curves, -1, self.dim).squeeze()
        us = us.reshape(self.n_curves, -1, self.dim).squeeze()
        ts = ts.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return grad(self.inner_product, argnums=0)(ts,zs,us,self.M[0])
        else:
            return jnp.stack([grad(self.inner_product, 
                                   argnums=0)(ts[i],
                                              zs[i],
                                              us[i],
                                              self.M[i]) for i in range(self.n_curves)]).reshape(-1)
    
    def gs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        zs = zs.reshape(self.n_curves, -1, self.dim).squeeze()
        us = us.reshape(self.n_curves, -1, self.dim).squeeze()
        ts = ts.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return grad(self.inner_product, argnums=1)(ts,zs,us,self.M[0])
        else:
            return jnp.stack([grad(self.inner_product, 
                                   argnums=1)(ts[i],
                                              zs[i],
                                              us[i],
                                              self.M[i]) for i in range(self.n_curves)]).reshape(-1, self.dim)
    
    def hs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        if self.n_tacks == 0:
            
            zs = zs.reshape(self.n_curves, -1, self.dim).squeeze()
            us = us.reshape(self.n_curves, -1, self.dim).squeeze()
            ts = ts.reshape(self.n_curves, -1).squeeze()
            
            return grad(self.inner_product_h, argnums=3)(ts,zs,us,us,self.M[0])
        else:
            
            t0, z0, u0 = ts[0], zs[0], us[0]
            
            zs = zs[1:].reshape(self.n_curves, self.T-1, -1).squeeze()
            us = us[1:].reshape(self.n_curves, self.T-1, -1).squeeze()
            ts = ts[1:].reshape(self.n_curves, -1).squeeze()
            
            val1 = [grad(self.inner_product_h, argnums=3)(jnp.hstack((t0, ts[0])),
                                                          jnp.vstack((z0.reshape(-1,self.dim), zs[0])),
                                                          jnp.vstack((u0.reshape(-1,self.dim), us[0])),
                                                          jnp.vstack((u0.reshape(-1,self.dim), us[0])),
                                                          self.M[0],
                                                          ).reshape(-1, self.dim)]
            
            for i in range(1,self.n_curves):
                
                val1.append(grad(self.inner_product_h, argnums=3)(ts[i],
                                                                  zs[i],
                                                                  us[i],
                                                                  us[i],
                                                                  self.M[i]).reshape(-1,self.dim))
            
            return jnp.concatenate(val1, axis=0).reshape(-1,self.dim)
                                              
    def gs_inv(self,
               ts:Array,
               zs:Array,
               us:Array,
               )->Array:
        
        zs = zs.reshape(self.n_curves, self.T-1, -1).squeeze()
        us = us.reshape(self.n_curves, self.T-1, -1).squeeze()
        ts = ts.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return vmap(self.M[0].Ginv)(ts,zs,us)
        else:
            return jnp.stack([vmap(self.M[i].Ginv)(ts[i],
                                                   zs[i],
                                                   us[i]) for i in range(self.n_curves)]).reshape(-1,
                                                                                                 self.dim,
                                                                                                 self.dim)
                                                                                                  
    def update_ts(self,
                  zs:Array,
                  us:Array,
                  )->Array:
        
        zs = zs.reshape(self.n_curves, self.T-1, -1)

        time_curves = []
        ts = self.update_time_path(self.t0, self.z0, zs[0], zs[1][0], self.M[0])
        if self.n_tacks < 1:
            return ts.reshape(-1)
        else:
            time_curves.append(ts)
            for i in range(1,self.n_curves-1):
                ts = self.update_time_path(ts[-1], zs[i][0], zs[i][1:], zs[i+1][0], self.M[i])
                time_curves.append(ts)
    
            ts = self.update_time_path(ts[-1], zs[-1][0], zs[-1][1:], self.zT, self.M[self.n_tacks])
            time_curves.append(ts)
            
            return jnp.hstack(time_curves).reshape(-1)
                                                                                                  
    def update_time_path(self,
                         t0:Array,
                         z0:Array,
                         zs:Array,
                         zT:Array,
                         M:LorentzFinslerManifold,
                         )->Array:
        
        def step(t:Array,
                 step:Tuple[Array,Array],
                 )->Array:
            
            z, dz = step
            
            t += M.F(t, z, dz)
            
            return (t,)*2
        
        zs = jnp.vstack((z0, zs))
        us = jnp.vstack((zs[1:]-zs[:-1],
                         zT-zs[-1]
                         ))
        
        _, ts = lax.scan(step,
                         init=t0,
                         xs = (zs, us),
                         )
        
        return ts
                                              
    def update_zs(self,
                  zs:Array,
                  alpha:Array,
                  us_hat:Array,
                  us:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*us_hat[:-1]+(1-alpha)*us[:-1], axis=0)
    
    def update_us(self,
                  gs_inv:Array,
                  mus:Array,
                  )->Array:
        
        return -0.5*jnp.einsum('tij,tj->ti', gs_inv, mus)
    
    def pi(self,
           rs:Array,
           Lts:Array,
           )->Array:
        
        def step(pis:Array,
                 step:Tuple[Array,Array],
                 )->Tuple[Array, Array]:
            
            rs, Ls = step
            
            return ((rs+pis*Ls+pis),)*2
        
        _, pi = lax.scan(step,
                         xs=(rs[::-1], Lts[::-1]),
                         init=0.0,
                         )
        
        return jnp.hstack((pi[::-1], 0.0)).reshape(-1,1)
    
    def unconstrained_opt(self, 
                          rs:Array,
                          hs:Array, 
                          gs:Array, 
                          gs_inv:Array,
                          pis:Array,
                          Lts:Array,
                          Lzs:Array,
                          Lus:Array,
                          )->Array:
        
        g_cumsum = jnp.vstack((jnp.cumsum((gs+pis[1:]*Lzs)[::-1], axis=0)[::-1], jnp.zeros((1,self.dim))))
        ginv_sum = jnp.sum(gs_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gs_inv, g_cumsum+hs+pis*Lus), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mus = muT+g_cumsum+hs+pis*Lus
        
        return mus
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = carry
        
        Lts = self.Lt(ts[:-1],zs,us[1:])
        Lzs = self.Lz(ts[:-1],zs,us[1:])
        Lus = jnp.vstack((jacfwd(self.M[0].F, argnums=2)(self.t0, self.z0, us[0]).reshape(-1,self.dim),
                          self.Lu(ts[:-1],zs,us[1:]),
                          ))

        pis = self.pi(rs, Lts)

        mus = self.unconstrained_opt(rs, hs, gs, gs_inv,
                                     pis, Lts, Lzs, Lus)

        us_hat = self.update_us(gs_inv, mus)
        tau = self.line_search(zs, us_hat, us)

        us = tau*us_hat+(1.-tau)*us
        zs = self.z0+jnp.cumsum(us[:-1], axis=0)
        ts = self.update_ts(zs, us)

        rs = self.rs(ts[:-1], zs, us[1:])
        hs = self.hs(jnp.hstack((self.t0, ts[:-1])), jnp.vstack((self.z0.reshape(1,-1), zs)), us)
        gs = self.gs(ts[:-1], zs, us[1:])
        gs_inv = jnp.vstack((self.M[0].Ginv(self.t0, self.z0, us[0]).reshape(-1,self.dim,self.dim), 
                             self.gs_inv(ts[:-1],zs,us[1:])))
        grad = self.Denergy(zs)
        
        return (ts, zs, us, rs, hs, gs, gs_inv, grad, idx+1)
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 n_tacks:int=0,
                 )->Array:
        
        t0 = t0.astype("float64")
        z0 = z0.astype("float64")
        zT = zT.astype("float64")
        
        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        self.n_tacks = n_tacks
        self.n_curves = n_tacks+1

        self.diff = zT-z0
        
        self.dtype = z0.dtype
        self.dim = len(z0)
        
        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_zs,
                                        grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                        **self.line_search_params,
                                        )

        zs = self.init_curve()
        us = jnp.ones((self.T+(self.T-1)*self.n_tacks, self.dim), dtype=self.dtype)*self.diff/self.T
        ts = self.update_ts(zs, us)
        
        rs = self.rs(ts[:-1], zs, us[1:])
        hs = self.hs(jnp.hstack((self.t0, ts[:-1])), jnp.vstack((self.z0.reshape(1,-1), zs)), us)
        gs = self.gs(ts[:-1], zs, us[1:])

        gs_inv = jnp.vstack((self.M[0].Ginv(self.t0, self.z0, us[0]).reshape(-1,self.dim,self.dim), 
                             self.gs_inv(ts[:-1],zs,us[1:])))
        grad = self.Denergy(zs)
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = lax.while_loop(self.cond_fun, 
                                                                   self.while_step, 
                                                                   init_val=(ts, zs, us, rs, hs, gs, gs_inv, grad, 0))
        
        ts = jnp.hstack((self.t0, ts))
        zs = jnp.vstack((z0, zs, zT))
            
        return ts, zs, grad, idx
