#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:46:22 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jscipy
from jax import vmap, lax, Array

from scipy.optimize import minimize

from typing import Callable

#%% Brownian

class dWs(object):
    
    def __init__(self,
                 dim:int = 1,
                 n_points:int=100,
                 seed:int=2712
                 )->None:
        
        self.dim = dim
        self.seed = seed
        self.key = jrandom.PRNGKey(self.seed)
        self.n_points=n_points
        
        return 
    
    def __str__(self)->str:
        
        return "Brownian Motion generator"
    
    def trapez_rule(self, f_fun:Callable, t:Array)->Array:
        
        dt = jnp.diff(t)
        if callable(f_fun):
            fval = vmap(f_fun)(t).squeeze()
        else:
            fval = f_fun
    
        return jnp.concatenate((jnp.zeros((1, self.dim)), 
                                .5*jnp.cumsum((dt*(fval[1:]+fval[:-1]).T).T.reshape(-1, self.dim), axis=0)),
                               axis=0)
    
    def dWs(self, dt:Array=None, N_sim:int=1)->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = subkey
        
        if dt is None:
            _, dt = self.dts()
        
        x = jrandom.normal(subkey, shape=(N_sim, self.dim, len(dt)))
        
        return jnp.transpose(jnp.sqrt(dt)*x, axes=(0,2,1)).squeeze()
    
    def dts(self, t0:float=.0, T:float=1.0)->Array:
        
        t = jnp.linspace(t0,T,self.n_points, endpoint=True)
        
        return t, jnp.diff(t)

#%% Inference of time only dependent diffusion processes

class DiffusionTime(dWs):
    
    def __init__(self,
                 mu_fun:Callable,
                 sigma_fun:Callable,
                 dim:int = 1,
                 n_points:int=100,
                 seed:int=2712,
                 max_iter:int=100,
                 )->None:
        super().__init__(dim, n_points, seed)
        
        self.mu_fun = mu_fun
        self.sigma_fun = sigma_fun
        self.dim = dim
        self.max_iter = max_iter
        
        self.M_fun = lambda t,*theta: self.trapez_rule(lambda t: self.mu_fun(t,*theta), jnp.linspace(0,t,n_points))
        self.S_fun = lambda t, *theta: .5*self.trapez_rule(lambda t: self.sigma_fun(t,*theta)**2, jnp.linspace(0,t,n_points))
            
        return
    
    def __str__(self)->str:
        
        return "Time-only dependent drift and diffusion Ito process"
    
    def sim(self, x0:Array, t0:float=.0, T:float=1.0, N_sim:int=10, *theta)->Array:
        
        def step(carry, y):
            
            t, x = carry
            dt, dW = y
            
            t += dt
            x += (self.mu_fun(t, *theta)*dt+jnp.dot(self.sigma_fun(t, *theta), dW)).squeeze()
            
            return (t,x), x
        
        t, dt = self.dts(t0, T)
        dW = self.dWs(dt, N_sim)
        
        if N_sim == 1:
            Xt = lax.scan(step, init=(0,x0), xs=(dt, dW))[-1]
            
            return (t,jnp.concatenate((x0+jnp.zeros((1, self.dim)),
                                    Xt.reshape(self.n_points-1,self.dim)), 
                                    axis=0).squeeze())
        else:
            Xt = vmap(lambda dW: lax.scan(step, init=(0,x0), xs=(dt, dW))[-1])(dW)
            
            return (t,jnp.concatenate((x0+jnp.zeros((N_sim, 1, self.dim)),
                                    Xt.reshape(N_sim, self.n_points-1, self.dim)), 
                                    axis=1).squeeze())
        
    def mu(self, x0:Array, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, x0+self.trapez_rule(lambda t: self.mu_fun(t,*theta), t)
    
    def var(self, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, self.trapez_rule(lambda t: self.sigma_fun(t,*theta)**2, t)
        
    def pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St))
        else:
            val = vmap(lambda Mt, St, x0: jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St)))(Mt, St, x0)
            
            return jnp.prod(val)
        
    def log_pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def max_likelihood(self, X:Array, t:Array, x0:Array, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: vmap(lambda x,t: self.log_pdf(x,x0,t,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    def integral_mu(self, w0:Array, x0_fun:Callable, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        return Mt+x0+w0*T
    
    def integral_var(self, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        
        return St
    
    def integral_log_pdf(self, x:Array, x0_fun:Callable, t:Array, dim:int=2, *theta)->Array:
        
        t_grid = jnp.linspace(0,t,self.n_points).squeeze()        
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        if self.dim == 1:
            return -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def integral_max_likelihood(self, X:Array, t:Array, x0:Callable, dim, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: 
                                  vmap(lambda x,t: self.integral_log_pdf(x,x0, t,dim,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
#%% Diffusion Bridge

#https://perso.univ-rennes1.fr/bernard.delyon/2006simucond.pdf

class DiffusionBridgeTime(dWs):
    
    def __init__(self,
                 mu_fun:Callable,
                 sigma_fun:Callable,
                 T:float=1.0,
                 dim:int = 1,
                 n_points:int=100,
                 seed:int=2712,
                 max_iter:int=100,
                 )->None:
        super().__init__(dim, n_points, seed)
        
        self.mu_fun = mu_fun
        self.sigma_fun = sigma_fun
        self.dim = dim
        self.max_iter = max_iter
        self.T = T
        
        self.M_fun = lambda t,*theta: self.trapez_rule(lambda t: self.mu_fun(t,*theta), jnp.linspace(0,t,n_points))
        self.S_fun = lambda t, *theta: .5*self.trapez_rule(lambda t: self.sigma_fun(t,*theta)**2, jnp.linspace(0,t,n_points))
            
        return
    
    def __str__(self)->str:
        
        return "Time-only dependent drift and diffusion Ito process"
    
    def Pt(self, 
           A:Array,
           t:Array
           )->Array:
        
        return jscipy.linalg.expm(A*t)
    
    def Rst(self,
            s:Array,
            t:Array):
        
        T = jnp.minimum(s,t)
        u_grid = jnp.linspace(0,T,self.n_points)
        sigma_u = vmap(lambda t: self.sigma_fun(t))(u_grid)
        Pu = vmap(lambda t: self.Pt(self.A, t))(u_grid)
        
        Ps = self.Pt(self.A, s)
        Pt = self.Pt(self.A, t)

        int_val = vmap(lambda P, sigma: jnp.linalg.solve(P, sigma))(Pu, sigma_u)
        int_val = jnp.einsum('...ij,...ik->...jk', int_val, int_val)
        
        return jnp.matmul(Ps, jnp.matmul(self.trapez_rule(int_val, u_grid), Pt.T))
    
    def bridge_pdf(self, t:Array):
        
        return
    
    def sim(self, x0:Array, t0:float=.0, T:float=1.0, N_sim:int=10, *theta)->Array:
        
        def step(carry, y):
            
            t, x = carry
            dt, dW = y
            
            t += dt
            x += (self.mu_fun(t, *theta)*dt+jnp.dot(self.sigma_fun(t, *theta), dW)).squeeze()
            
            return (t,x), x
        
        t, dt = self.dts(t0, T)
        dW = self.dWs(dt, N_sim)
        
        if N_sim == 1:
            Xt = lax.scan(step, init=(0,x0), xs=(dt, dW))[-1]
            
            return (t,jnp.concatenate((x0+jnp.zeros((1, self.dim)),
                                    Xt.reshape(self.n_points-1,self.dim)), 
                                    axis=0).squeeze())
        else:
            Xt = vmap(lambda dW: lax.scan(step, init=(0,x0), xs=(dt, dW))[-1])(dW)
            
            return (t,jnp.concatenate((x0+jnp.zeros((N_sim, 1, self.dim)),
                                    Xt.reshape(N_sim, self.n_points-1, self.dim)), 
                                    axis=1).squeeze())
        
    def mu(self, x0:Array, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, x0+self.trapez_rule(lambda t: self.mu_fun(t,*theta), t)
    
    def var(self, T:float=1.0, *theta)->Array:
        
        t = jnp.linspace(0,T,self.n_points)
        
        return t, self.trapez_rule(lambda t: self.sigma_fun(t,*theta)**2, t)
        
    def pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St))
        else:
            val = vmap(lambda Mt, St, x0: jnp.exp(-.25*((x-x0-Mt)**2)/St)/(2*jnp.sqrt(jnp.pi*St)))(Mt, St, x0)
            
            return jnp.prod(val)
        
    def log_pdf(self, x:Array, x0:Array, t:Array, *theta)->Array:
        
        Mt = self.M_fun(t,*theta)[-1]
        St = self.S_fun(t,*theta)[-1]
        
        if self.dim == 1:
            return -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def max_likelihood(self, X:Array, t:Array, x0:Array, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: vmap(lambda x,t: self.log_pdf(x,x0,t,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    def integral_mu(self, w0:Array, x0_fun:Callable, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        return Mt+x0+w0*T
    
    def integral_var(self, T:float=1.0, *theta)->Array:
        
        t_grid = jnp.linspace(0,T,self.n_points).squeeze()
        
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        
        return St
    
    def integral_log_pdf(self, x:Array, x0_fun:Callable, t:Array, dim:int=2, *theta)->Array:
        
        t_grid = jnp.linspace(0,t,self.n_points).squeeze()        
        
        Mt = self.trapez_rule(vmap(lambda t: self.M_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        St = self.trapez_rule(vmap(lambda t: self.S_fun(t,*theta)[-1])(t_grid), t_grid)[-1]
        x0 = x0_fun(t_grid, *theta)
        
        if self.dim == 1:
            return -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St)
        else:
            val = vmap(lambda Mt, St, x0: -.25*((x-t*x0-Mt)**2)/St-.5*jnp.log(4*jnp.pi*St))(Mt, St, x0)
            
            return jnp.sum(val)
        
    def integral_max_likelihood(self, X:Array, t:Array, x0:Callable, dim, theta_init:Array)->Array:
        
        def loss_fun(theta:Array):
            
            return -jnp.mean(vmap(lambda y,s: 
                                  vmap(lambda x,t: self.integral_log_pdf(x,x0, t,dim,theta))(y,s))(X,t))
        
        
        sol = minimize(loss_fun, theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
        
    
        
        
        
        
        
        
        
        
    
    