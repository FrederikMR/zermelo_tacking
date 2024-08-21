#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.lorentz_finsler.manifolds import LorentzFinslerManifold
from geometry.line_search import Backtracking, Bisection

#%% Gradient Descent Estimation of Geodesics

class GEORCE(ABC):
    def __init__(self,
                 M:LorentzFinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_method:str="soft",
                 line_search_params:Dict = {},
                 obj_method:str="tensor",
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        if line_search_method in ['soft', 'exact']:
            self.line_search_method = line_search_method
        else:
            raise ValueError(f"Invalid value for line search method, {line_search_method}")
            
        if not (obj_method in ['tensor', 'finsler']):
                raise ValueError(f"The obj_method should be either tensor or finsler. Not {obj_method}.")
        else:
            if obj_method == 'tensor':
                self.obj_fun = lambda t,z,u: M.g(t,z,u)
            else:
                self.obj_fun = lambda t,z,u: M.F(t,z,u)**2

        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   self.T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def time_update(self,
                    z:Array,
                    dt:Array,
                    )->Array:
        
        def euler_step(t:Array,
                       step:Tuple[Array, Array],
                       )->Array:

            z, dz = step

            t += self.M.F(t,z,dz/self.dt)*self.dt
            
            return (t,)*2
        
        dz = z[1:]-z[:-1]
        dz = jnp.vstack((z[0]-self.z0, dz, self.zT-z[-1]))
        z = jnp.vstack((self.z0, z))
        
        _, t = lax.scan(euler_step,
                        init=0.0,
                        xs=(z,dz),
                        )
        
        return t
    
    def energy(self, 
               zt:Array,
               t:Array,
               *args
               )->Array:
        
        term1 = zt[0]-self.z0
        val1 = self.obj_fun(0.0,self.z0, term1)
        
        term2 = zt[1:]-zt[:-1]
        val2 = vmap(lambda t,x,v: self.obj_fun(t,x,v))(t[:-2], zt[:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = self.obj_fun(t[-2],zt[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Denergy(self,
                t:Array,
                zt:Array,
                )->Array:
        
        return grad(self.energy, argnums=0)(zt,t)
    
    def inner_product(self,
                      t:Array,
                      zt:Array,
                      ut:Array,
                      )->Array:
        
        Gt = vmap(self.M.G, in_axes=(0, 0, 0))(t,zt,ut)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def gt(self,
           t:Array,
           zt:Array,
           ut:Array,
           )->Array:
        
        return grad(self.inner_product, argnums=1)(t,zt,ut)
    
    def update_xt(self,
                  zt:Array,
                  alpha:Array,
                  t:Array,
                  ut_hat:Array,
                  ut:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*ut_hat[:-1]+(1-alpha)*ut[:-1], axis=0)
    
    def unconstrained_opt(self, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array,Array, Array, int],
                 )->Array:
        
        t, zt, ut, gt, gt_inv, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array,Array, Array, int],
                     )->Array:
        
        t, zt, ut, gt, gt_inv, grad, idx = carry
        
        mut = self.unconstrained_opt(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, t, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        t = self.time_update(zt, self.dt)

        gt = self.gt(t[:-1],zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.M.Ginv(0.0,self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.M.Ginv)(t[:-1],zt,ut[1:])))
        grad = self.Denergy(t,zt)
        
        return (t, zt, ut, gt, gt_inv, grad, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array,Array],
                 idx:int,
                 )->Array:
        
        t, zt, ut = carry
        
        gt = self.gt(t[:-1],zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        gt_inv = jnp.vstack((self.M.Ginv(0.0,self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.M.Ginv)(t[:-1],zt,ut[1:])))
        
        mut = self.unconstrained_opt(gt, gt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        tau = self.line_search(zt, t, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        t = self.time_update(zt, self.dt)

        return ((t, zt, ut),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        dtype = z0.dtype
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        self.dt = 1.0/self.T
        
        if self.line_search_method == "soft":
            self.line_search = Backtracking(obj_fun=self.energy,
                                            update_fun=self.update_xt,
                                            grad_fun = lambda z,*args: self.Denergy(args[0],z).reshape(-1),
                                            **self.line_search_params,
                                            )
        else:
            self.line_search = Bisection(obj_fun=self.energy, 
                                         update_fun=self.update_xt,
                                         **self.line_search_params,
                                         )
        
        self.diff = zT-z0
        ut = jnp.ones((self.T, self.dim), dtype=dtype)*self.diff/self.T
        
        self.z0 = z0
        self.zT = zT
        
        if step == "while":
            t = self.time_update(zt, self.dt)
            gt = self.gt(t[:-1],zt,ut[1:])#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
            gt_inv = jnp.vstack((self.M.Ginv(0.0,self.z0, ut[0]).reshape(-1,self.dim,self.dim), 
                                 vmap(self.M.Ginv)(t[:-1],zt,ut[1:])))
            grad = self.Denergy(t, zt)
            
            t, zt, _, _, _, grad, idx = lax.while_loop(self.cond_fun, 
                                                    self.while_step, 
                                                    init_val=(t,zt, ut, gt, gt_inv, grad, 0))
            
            zt = jnp.vstack((z0, zt, zT))
            t = jnp.hstack((0.0, t))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(t,zt, ut),
                              xs=jnp.ones(self.max_iter),
                              )
            
            t = val[0]
            zt = val[1]
            grad = vmap(self.Denergy)(t, zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return t, zt, grad, idx

        