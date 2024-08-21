#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:53:14 2024

@author: fmry
"""

#%% Modules

from geometry.setup import *

from geometry.manifolds import LorentzFinslerManifold
from .bisection import Bisection
from .backtracking import Backtracking

#%% GEORCE Estimation of Tack Points and Geodesics

class GEORCE(ABC):
    def __init__(self,
                 M:List[LorentzFinslerManifold],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_method:str="soft",
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        if line_search_method in ['soft', 'exact']:
            self.line_search_method = line_search_method
        else:
            raise ValueError(f"Invalid value for line search method, {line_search_method}")

        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype).reshape(-1,1)+z0
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
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
    
    def time_fun(self,
                 zt:Array,
                 )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()
        
        if self.n_tacks < 1.:
            t = self.time_integral(self.t0,
                                   jnp.vstack((self.z0, zt)),
                                   self.zT,
                                   self.M[0],
                                   )[:-1]
        elif self.n_tacks <2.:
            t1 = self.time_integral(self.t0,
                                    jnp.vstack((self.z0, zt[0])),
                                    zt[1][0],
                                    self.M[0],
                                    )
            
            tT = self.time_integral(t1[-1],
                                    zt[-1],
                                    self.T,
                                    self.M[self.n_tacks],
                                    )
            
            t = jnp.hstack((t1, tT[:-1]))
        else:
            t1 = self.time_integral(self.t0,
                                    jnp.vstack((self.z0, zt[0])),
                                    zt[1][0],
                                    self.M[0],
                                    )
            t0 = t1[-1]
            times = []
            for i in range(1,self.n_tacks):
                t = self.time_integral(t0,
                                       zt[i],
                                       zt[i+1][0],
                                       self.M[i],
                                       )
                t0 = t[-1]
                times.append(t)
            tT = self.time_integral(t0,
                                    zt[-1],
                                    self.T,
                                    self.M[self.n_tacks],
                                    )
            
            t = jnp.hstack((t1, jnp.stack(times).reshape(-1), tT[:-1]))
        
        return t
    
    def time_integral(self,
                      t0:Array,
                      zt:Array,
                      zT:Array,
                      M:LorentzFinslerManifold,
                      )->Array:
        
        def time_update(t:Array,
                        step:Tuple[Array,Array],
                        )->Array:
            
            z, dz = step
            
            t += M.F(t, z, dz/self.dt)*self.dt
            
            return (t,)*2

        dz = jnp.vstack((zt[1:]-zt[:-1], zT-zt[-1]))
        _, t = lax.scan(time_update,
                        init=t0,
                        xs = (zt, dz),
                        )
        
        return t
    
    def energy(self, 
               t:Array,
               zt:Array, 
               *args
               )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()
        t = t.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks < 1.:
            energy = self.path_energy(jnp.hstack((self.t0, t)), 
                                      jnp.vstack((self.z0, zt)), 
                                      self.zT, 
                                      self.M[0],
                                      *args)
            
            return energy
        
        elif self.n_tacks < 2.:
            e1 = self.path_energy(jnp.hstack((self.t0, t[0])),
                                  jnp.vstack((self.z0, zt[0])),
                                  zt[1][0],
                                  self.M[0],
                                  *args,
                                  )
            
            eT = self.path_energy(t[-1],
                                  zt[-1],
                                  self.zT,
                                  self.M[self.n_tacks],
                                  *args,
                                  )
            
            return e1+eT
            
        else:
            e1 = self.path_energy(jnp.hstack((self.t0, t[0])),
                                  jnp.vstack((self.z0, zt[0])),
                                  zt[1][0],
                                  self.M[0],
                                  *args,
                                  )
            energy = []
            for i in range(1,self.n_tacks):
                e = self.path_energy(t[i],
                                     zt[i],
                                     zt[i+1][0],
                                     self.M[i],
                                     *args,
                                     )
                energy.append(e)
            eT = self.path_energy(t[-1],
                                  zt[-1],
                                  self.zT,
                                  self.M[self.n_tacks],
                                  *args,
                                  )
            
            return e1+jnp.sum(jnp.stack(energy))+eT
    
    def path_energy(self,
                    t:Array,
                    zt:Array,
                    zT:Array,
                    M:LorentzFinslerManifold,
                    *args,
                    )->Array:
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda t,z,u: M.F(t,z,u)**2)(t[:-1], zt[:-1], term2)
        
        term3 = zT-zt[-1]
        val3 = M.F(t[-1],zt[-1],term3)**2
        
        return jnp.sum(val2)+val3
    
    def Denergy(self,
                t:Array,
                zt:Array,
                )->Array:
        
        return grad(self.energy, argnums=1)(t, zt)
    
    def inner_product(self,
                      t:Array,
                      zt:Array,
                      ut:Array,
                      M:LorentzFinslerManifold,
                      )->Array:
        
        Gt = vmap(M.G, in_axes=(0,0,0))(t,zt,ut)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def inner_product_h(self,
                        t:Array,
                        zt:Array,
                        u0:Array,
                        ut:Array,
                        M:LorentzFinslerManifold,
                        )->Array:
        
        Gt = vmap(M.G, in_axes=(0,0,0))(t,zt,ut)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', u0, Gt, u0))
    
    def gt(self,
           t:Array,
           zt:Array,
           ut:Array,
           )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()
        ut = ut.reshape(self.n_curves, self.T-1, -1).squeeze()
        t = t.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return grad(self.inner_product, argnums=1)(t,zt,ut,self.M[0])
        else:
            return jnp.stack([grad(self.inner_product, 
                                   argnums=1)(t[i],
                                              zt[i],
                                              ut[i],
                                              self.M[i]) for i in range(self.n_curves)]).reshape(-1, self.dim)
    
    def ht(self,
           t:Array,
           zt:Array,
           ut:Array,
           )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()
        ut = ut.reshape(self.n_curves, self.T-1, -1).squeeze()
        t = t.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return grad(self.inner_product_h, argnums=3)(t,zt,ut,ut,self.M[0])
        else:
            return jnp.stack([grad(self.inner_product_h, 
                                   argnums=3)(t[i],
                                              zt[i],
                                              ut[i],
                                              ut[i],
                                              self.M[i]) for i in range(self.n_curves)]).reshape(-1, self.dim)
                                              
    def gt_inv(self,
               t:Array,
               zt:Array,
               ut:Array,
               )->Array:
        
        zt = zt.reshape(self.n_curves, self.T-1, -1).squeeze()
        ut = ut.reshape(self.n_curves, self.T-1, -1).squeeze()
        t = t.reshape(self.n_curves, -1).squeeze()
        
        if self.n_tacks == 0:
            return vmap(self.M[0].Ginv)(t,zt,ut)
        else:
            return jnp.stack([vmap(self.M[i].Ginv)(t[i],
                                                   zt[i],
                                                   ut[i]) for i in range(self.n_curves)]).reshape(-1,
                                                                                                  self.dim,
                                                                                                  self.dim)
                                              
    def update_xt(self,
                  t:Array,
                  zt:Array,
                  alpha:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1-alpha)*ut[:-1], axis=0)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        t, zt, ut, ht, gt, gt_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        t, zt, ut, ht, gt, gt_inv, grad, idx = carry
        
        mut = self.unconstrained_opt(ht, gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(t, zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        gt = self.gt(t,zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        ht = self.ht(t,zt,ut[:-1])
        gt_inv = jnp.vstack((self.M[0].Ginv(self.t0, self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             self.gt_inv(t, zt, ut[1:])))
        grad = self.Denergy(t, zt)
        
        return (t, zt, ut, ht, gt, gt_inv, grad, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        t, zt, ut = carry
        
        gt = self.gt(t, zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        ht = self.ht(t,zt,ut[:-1])
        gt_inv = jnp.vstack((self.M[0].Ginv(self.t0, self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             self.gt_inv(t, zt, ut[1:])))
        
        mut = self.unconstrained_opt(ht, gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(t, zt, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)

        return ((t, zt, ut),)*2
    
    def unconstrained_opt(self, 
                          ht:Array,
                          gt:Array, 
                          gt_inv:Array
                          )->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum+ht), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum+ht, muT))
        
        return mut
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 n_tacks:int=0,
                 step:str="while",
                 )->Array:
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_xt,
                                            grad_fun = lambda t,z,*args: self.Denergy(t,z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_xt,
                                         **self.line_search_params,
                                         )
            
        self.n_tacks = n_tacks
        self.n_curves = n_tacks+1
        
        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        
        self.dt = 1.0/(self.T+self.n_tacks*(self.T-1))
        self.diff = zT-z0
        
        self.dtype = z0.dtype
        self.dim = len(z0)
        
        zt = self.init_curve()
        t = self.time_fun(zt)
        ut = jnp.ones((self.T+(self.T-1)*self.n_tacks, self.dim), dtype=self.dtype)*self.diff/self.T
        
        if step == "while":
            
            gt = self.gt(t,zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            ht = self.ht(t,zt,ut[:-1])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gt_inv = jnp.vstack((self.M[0].Ginv(self.t0, self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                                 self.gt_inv(t, zt, ut[1:])))
            grad = self.Denergy(t, zt)
            
            t, zt, _, _, _, _, grad, idx = lax.while_loop(self.cond_fun, 
                                                          self.while_step, 
                                                          init_val=(t, zt, ut, ht, gt, gt_inv, grad, 0))
            
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
            t = self.time_fun(zt)
                
            _, val = lax.scan(self.for_step,
                              init=(t, zt, ut),
                              xs=jnp.ones(self.max_iter),
                              )
            
            t, zt = val[0], val[1]
            grad = vmap(self.Denergy)(t, zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return t, zt, grad, idx
