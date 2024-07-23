#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:07:23 2022

@author: frederik
"""

#%% Sources

#http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf

#%% Modules

import jax.numpy as jnp
from jax import grad, jacrev, jacfwd, vmap, jit, lax, Array, jacrev
#For double precision
from jax.config import config

from scipy.optimize import minimize

#JAX Optimization
from jax.example_libraries import optimizers

import gp.kernels as km
import gp.sp as sp
import gp.ode_integrator as oi

from typing import Tuple, Callable

#%% manifold class

class riemannian_manifold(object):
    
    def __init__(self):
        
        self.G = None
        self.chris = None

#%% GP class

class gp_features(object):
    
    def __init__(self):
        
        self.G = None
        self.J = None
        
#%% Gaussian Process

class GP(object):
    def __init__(self,
                 mu_fun=None,
                 k_fun=None,
                 optimize:bool=False,
                 max_iter:int=1000,
                 delta_stable:float=1e-10,
                 )->None:
        
        if mu_fun is None:
            self.mu_fun = lambda x: jnp.zeros(len(x))
        else:
            self.mu_fun = mu_fun
            
        if k_fun is None:
            self.k_fun = km.gaussian_kernel
            self.f_fun = km.gaussian_kernel
        else:
            self.k_fun = k_fun
            self.f_fun = k_fun
            
        self.optimize=optimize
        self.max_iter = max_iter
        self.delta_stable=delta_stable
        self.theta = None
            
        self.Dm_fun = jacfwd(self.mu_fun, argnums=0)
        self.DDm_fun = jacfwd(self.Dm_fun, argnums=0)
        self.DDDm_fun = jacrev(self.DDm_fun, argnums=0)

        self.Dk1_fun = jacfwd(self.k_fun, argnums=0)
        self.Dk2_fun = jacfwd(self.k_fun, argnums=1)
        self.DDk_fun = jacfwd(jacrev(self.k_fun, argnums=0), argnums=1)
        self.DDDk_fun = jacfwd(jacfwd(jacrev(self.k_fun, argnums=0), argnums=1), argnums=0)
        self.DDDDk_fun = jacfwd(jacfwd(jacfwd(jacrev(self.k_fun, argnums=0), argnums=1), argnums=0), argnums=1)
            
        return
    
    def __str__(self)->str:
        
        return "Gaussian Process fitting object"
    
    def sim_prior(self, X_test:Array, N_sim:int=10)->Array:
        
        if X_test.ndim==1:
            X_test = X_test.reshape(1,-1)
            
        _, N_data = X_test.shape
        
        if X_test.ndim == 1:
            mu = self.mu_fun(X_test)
        else:
            mu = vmap(self.mu_fun)(X_test)
        
        K = km.km(X_test.T, kernel_fun = self.k_fun)+jnp.eye(N_data)*self.delta_stable
        
        return sp.sim_multinormal(mu=mu, cov=K, dim=(N_sim,))
    
    def log_ml(self, theta:Array):

        K11 = self.K11_theta(theta)
        
        if self.N_obs == 1:
            pYX = -0.5*(self.y_training.dot(jnp.linalg.solve(K11, self.y_training)) \
                        +jnp.log(jnp.linalg.det(K11))+self.N_training*jnp.log(2*jnp.pi))
        else:
            pYX = vmap(lambda y: (self.y_training.dot(jnp.linalg.solve(K11, y)) \
                                  +jnp.log(jnp.linalg.det(K11))+self.N_training*jnp.log(2*jnp.pi)))(self.y_training)
            pYX = -0.5*jnp.sum(pYX)
             
        return pYX
    
    def optimize_hyper(self, theta_init:Array):
        
        #@jit
        #def update(carry:Tuple[Array, Array, object], idx:int
        #           )->Tuple[Tuple[Array, Array, object],
        #                    Tuple[Array, Array]]:
        #    
        #    mu, opt_state = carry
        #    
        #    #grad = jnp.clip(grad, min_step, max_step)
        #    
        #    grad_val = grad_fn(mu)
        #    #opt_state = opt_update(idx, grad_val, opt_state)
        #    #mu = get_params(opt_state)
        #    mu -= 0.01*grad_val
        #    
        #    return (mu, opt_state), mu
        #
        #grad_fn = grad(self.log_ml)
        #
        #opt_init, opt_update, get_params = optimizers.adam(0.1, b1=0.9, b2=0.999, eps=1e-8)
        #    
        #opt_state = opt_init(theta_init)
        #_, theta = lax.scan(update, init = (theta_init, opt_state), xs = jnp.arange(0,self.max_iter,1))
        #
        #return theta[-1]
        #
        #def grad_pYX(theta):
        #    
        #    K11 = K11_theta(theta)+jnp.eye(N_training)*delta_stable
        #    K11_inv = jnp.linalg.inv(K11)
        #    K_theta = grad_K(theta)
        #    
        #    alpha = jnp.linalg.solve(K11, y_training).reshape(-1,1)
        #    alpha_mat = alpha.dot(alpha.T)
        #                            
        #    return 0.5*jnp.trace((alpha_mat-K11_inv).dot(K_theta))

        #sol = minimize(lambda theta: -log_ml(theta), theta_init, jac=grad_pYX,
        #             method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False
        
        sol = minimize(lambda theta: -self.log_ml(theta), theta_init,
                     method='Nelder-Mead', options={'maxiter':self.max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    
    def fit(self, X_training:Array, y_training:Array, sigman:float=1.0, theta_init=None)->None:
        
        sigman2 = sigman**2
        self.y_training = y_training
        self.sigman2 = sigman
        self.K11_theta = lambda theta: km.km(X_training.T, X_training.T, lambda x,y: self.f_fun(x,y,*theta))+sigman2*jnp.eye(N_training)
        
        if X_training.ndim == 1:
            X_training = X_training.reshape(1,-1)
        else:
            X_training = X_training
            
        self.X_training = X_training
            
        dim, N_training = X_training.shape
        
        self.N_training = N_training
        
        if y_training.ndim == 1:
            self.N_obs = 1
        else:
            self.N_obs = y_training.shape[-1]
        
        if self.optimize:
            theta = self.optimize_hyper(theta_init)
            self.theta = theta
            self.k_fun = lambda x,y: self.f_fun(x,y,*theta)
        
        self.K11 = km.km(X_training.T, X_training.T, self.k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*self.delta_stable
        self.K11_inv = jnp.linalg.inv(self.K11)
            
        self.m_training = vmap(lambda x: self.mu_fun(x))(X_training.T).squeeze()
        
        return
        
    def posterior_dist(self, X_test:Array)->Tuple[Array, Array]:
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        N_test = X_test.shape[-1]
        m_test = vmap(lambda x: self.mu_fun(x))(X_test.T).squeeze()
        
        K21 = km.km(X_test.T , self.X_training.T, self.k_fun).squeeze()
        K12 = km.km(self.X_training.T, X_test.T, self.k_fun).squeeze()
        K22 = km.km(X_test.T, X_test.T, self.k_fun).squeeze()
        
        solved = jnp.linalg.solve(self.K11.T, K21.T).T

        if self.N_obs == 1:
            mu_post = m_test+(solved @ (self.y_training-self.m_training))
        else:
            mu_post = vmap(lambda y: m_test+(solved @ (y-self.m_training)))(self.y_training)
            
        cov_post = K22-(solved @ K12)
        
        return mu_post, cov_post+jnp.eye(N_test)*self.delta_stable
    
    def sim_posterior(self, X_test:Array, N_sim:int=10)->Array:
        
        mu_post, cov_post = self.posterior_dist(X_test)
        
        return sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=(N_sim,))
    
    def jacobian_mom(self, X_test:Array):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)
            
        N_test = X_test.shape[-1]
 
        Dm_test = vmap(lambda x: self.Dm_fun(x))(X_test.T).squeeze()    
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK1 = km.km(X_test.T, self.X_training.T, self.Dk1_fun).squeeze()
        DK2 = km.km(self.X_training.T, X_test.T, self.Dk2_fun).squeeze()
        DDK = km.km(X_test.T, X_test.T, self.DDk_fun).squeeze()
        
        solved = jnp.linalg.solve(self.K11.T, DK1.T).T
        
        if self.N_obs == 1:
            mu_post = Dm_test+(solved @ (self.y_training-self.m_training))
        else:
            mu_post = vmap(lambda y: Dm_test+(solved @ (y-self.m_training)))(self.y_training)
        
        cov_post = DDK-(solved @ DK2)
        
        return mu_post.T, cov_post+jnp.eye(N_test)*self.delta_stable

def gp(X_training, y_training, sigman = 1.0, m_fun = None, k = None, 
       theta_init = None, optimize=False, max_iter = 100, delta_stable=1e-10):
    
    def sim_prior(X, n_sim=10):
        
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        _, N_data = X.shape
        
        if m_fun is None:
            mu = jnp.zeros(N_data)
        else:
            mu = m_fun(X)
        
        K = km.km(X.T, kernel_fun = k_fun)+jnp.eye(N_data)*delta_stable
        
        return sp.sim_multinormal(mu=mu, cov=K, dim=n_sim)
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        N_test = X_test.shape[-1]
        m_test = m_fun(X_test)
        
        K21 = km.km(X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K21).T
        if N_obs == 1:
            mu_post = m_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: m_test+(solved @ (y-m_training)))(y_training)
            
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post+jnp.eye(N_test)*delta_stable
    
    def sim_post(X_test, n_sim=10):
        
        mu_post, cov_post = GP.post_mom(X_test)
        
        if N_obs == 1:
            gp = sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=n_sim)
        else:
            gp = vmap(lambda mu: sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=n_sim))(mu_post)
        
        return gp
    
    def log_ml(theta):

        K11 = K11_theta(theta)
        
        if N_obs == 1:
            pYX = -0.5*(y_training.dot(jnp.linalg.solve(K11, y_training))+jnp.log(jnp.linalg.det(K11))+N_training*jnp.log(2*jnp.pi))
        else:
            pYX = vmap(lambda y: (y_training.dot(jnp.linalg.solve(K11, y))+jnp.log(jnp.linalg.det(K11))+N_training*jnp.log(2*jnp.pi)))(y_training)
            pYX = -0.5*jnp.sum(pYX)
             
        return pYX
    
    def optimize_hyper(theta_init):

        #def grad_pYX(theta):
        #    
        #    K11 = K11_theta(theta)+jnp.eye(N_training)*delta_stable
        #    K11_inv = jnp.linalg.inv(K11)
        #    K_theta = grad_K(theta)
        #    
        #    alpha = jnp.linalg.solve(K11, y_training).reshape(-1,1)
        #    alpha_mat = alpha.dot(alpha.T)
        #                            
        #    return 0.5*jnp.trace((alpha_mat-K11_inv).dot(K_theta))

        #sol = minimize(lambda theta: -log_ml(theta), theta_init, jac=grad_pYX,
        #             method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False
        
        sol = minimize(lambda theta: -log_ml(theta), theta_init,
                     method='Nelder-Mead', options={'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
        
    dim, N_training = X_training.shape
    
    if y_training.ndim == 1:
        N_obs = 1
    else:
        N_obs = y_training.shape[-1]
    
    if m_fun is None:
        m_fun = lambda x: jnp.zeros(x.shape[-1])
        
    if k is None:
        k = km.gaussian_kernel
        
    K11_theta = lambda theta: km.km(X_training, X_training, lambda x,y: k(x,y,*theta))+sigman2*jnp.eye(N_training)
    
    if optimize:
        theta = optimize_hyper(theta_init)
        k_fun = lambda x,y: k(x,y,*theta)
    else:
        k_fun = k
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*delta_stable
        
    m_training = m_fun(X_training)
    
    GP = gp_features() 
    GP.sim_prior = sim_prior
    GP.post_mom = jit(post_mom)
    GP.sim_post = sim_post
    GP.log_ml = jit(log_ml)
    GP.opt = optimize_hyper
    
    return GP

#%% Riemannian manifold with expected metric

def RM_EG(X_training, y_training, sigman = 1.0, m_fun = None, Dm_fun = None, k_fun = None, Dk_fun = None, 
          DDk_fun = None, DDDk_fun = None, DDDDk_fun = None,
      theta_init = None, optimize=False, grad_K = None,  max_iter = 100, delta_stable=1e-10,
      grid=jnp.linspace(0,1,100), method='euler', tol=1e-05):
    
    def curvature_operator(x):
        
        Dchris = RM.Dchris(x)
        chris = RM.chris(x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x):
        
        CO = RM.CO(x)
        G = RM.G(x)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2):
        
        CT = RM.CT(x)[0,1,1,0]
        G = RM.G(x)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def chris_symbols(x):
                
        G_inv = RM.G_inv(x)
        DG = RM.DG(x)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def Dchris(x):
        
        DG = RM.DG(x)
        G, DDG = RM.DDG(x)
        G_inv, DG_inv = RM.DG_inv(x)
        
        Dchris1 = jnp.einsum('jlik, lm->mjik', DDG, G_inv) \
                    +jnp.einsum('lijk, lm->mjik', DDG, G_inv) \
                    -jnp.einsum('ijlk, lm->mjik', DDG, G_inv)
        Dchris2 = jnp.einsum('jli, lmk->mjik', DG, DG_inv) \
                    +jnp.einsum('lij, lmk->mjik', DG, DG_inv) \
                    -jnp.einsum('ijl, lmk->mjik', DG, DG_inv)
        
        return 0.5*(Dchris1+Dchris2)
    
    def ivp_geodesic(x,v):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))

        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def pt(v0, gamma, Dgamma, coef):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat, coef)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        fun = jit(eq_pt)
        v = oi.ode_integrator(v0, fun, grid = grid, method=method)
        
        return v
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)

        N_test = X_test.shape[-1]
        m_test = m_fun(X_test)
        
        K21 = km.km(X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K21).T
        if N_obs == 1:
            mu_post = m_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: m_test+(solved @ (y-m_training)))(y_training)
            
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post+jnp.eye(N_test)*delta_stable
    
    def jacobian_mom(X_test):

        X_test = X_test.reshape(-1)        
        Dm_test = Dm_fun(X_test.reshape(-1,1)).squeeze()    
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK = vmap(lambda x: Dk_fun(x, X_test))(X_training.T)
        DDK = DDk_fun(X_test, X_test)  
        
        solved = jnp.linalg.solve(K11, DK).T
        
        if N_obs == 1:
            mu_post = Dm_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: Dm_test+(solved @ (y-m_training)))(y_training)
        
        cov_post = DDK-(solved @ DK)
        
        return mu_post.T, cov_post+jnp.eye(dim)*delta_stable #SOMETHING WRONG COV IS NOT POSTIVE (SYMMETRIC DEFINITE), BUT IS NEGATIVE SYMMETRIC DEFINITE
    
    def Emmf(X_test):

        mu_post, cov_post = RM.jac_mom(X_test)
        
        if N_obs == 1:    
            mu_post = mu_post.reshape(1,-1)
        
        EG = mu_post.dot(mu_post.T)+N_obs*cov_post
        
        return EG
    
    def DEJ(X_test):
        
        def DEJ_single(y):
            
            y_diff = y-m_training
            
            term1 = jnp.einsum('ikm,i->km', solved2, y_diff)
            term2 = -jnp.einsum('jk,jl->lk', solved1, jnp.einsum('jmk,m->jk', solvedDK11, y_diff)+Dm_training)

            return DDm_test+term1+term2
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        
        DDm_test = DDm_fun(X_test.reshape(-1,1)).squeeze()
        
        if N_obs == 1:
            return DEJ_single(y_training)
        else:
            return jnp.transpose(vmap(DEJ_single)(y_training), axes=(1,0,2))
        
    def DDEJ(X_test):
        
        def DDEJ_single(y):
            
            y_diff = y-m_training
            
            term1 = jnp.einsum('jikm,j->ikm', solved3, y_diff)
            term2 = -2.0*jnp.einsum('jlik,j->lik', jnp.einsum('mik,mjl->jlik', solved2, solvedDK11), y_diff)
            term3 = -jnp.einsum('mik, mj->jik', solved2, Dm_training)
            term4 = 2.0*jnp.einsum('idlk,i->dlk', jnp.einsum('mlk,mid->idlk', jnp.einsum('jk,jml->mlk', solved1, solvedDK11), solvedDK11), y_diff)
            term5 = -jnp.einsum('mulk,m->ulk', jnp.einsum('jk,jmul->mulk', solved1, solvedDDK11), y_diff)
            term6 = -jnp.einsum('mik,mj->jik', solved2, Dm_training)
            term7 = -jnp.einsum('ik,iml->mlk', solved1, DDm_training)
            
            return DDDm+term1+term2+term3+term4+term5+term6+term7
        
        X_test = X_test.reshape(-1)
        DDDm = DDDm_fun(X_test.reshape(-1,1)).squeeze()
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK12 = vmap(lambda x: DDDk_fun(X_test, x))(X_training.T)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        solved3 = jnp.einsum('jikm, jl->likm', DDDK12, K11_inv)

        if N_obs == 1:
            return DDEJ_single(y_training)
        else:
            return jnp.einsum('ijkm->jikm', vmap(DDEJ_single)(y_training))
    
    def Dcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK = DDDk_fun(X_test, X_test)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        
        term1 = -jnp.einsum('mik,ml->lik', solved2, DK12)
        term2 = jnp.einsum('mdk,ml->ldk', jnp.einsum('jk,jmd->mdk', solved1, solvedDK11), DK12)
        term3 = -jnp.einsum('jk,jmd->mdk', solved1, DDK12)
        
        return DDDK+term1+term2+term3
    
    def DDcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK12 = vmap(lambda x: DDDk_fun(X_test, x))(X_training.T)
        DDDDK22 = DDDDk_fun(X_test, X_test)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        solved3 = jnp.einsum('jikm, jl->likm', DDDK12, K11_inv)
        
        term1 = -jnp.einsum('likm,ld->dikm', solved3, DK12)
        term2 = 2.0*jnp.einsum('jdik,ju->udik', jnp.einsum('mik, mjd->jdik', solved2, solvedDK11), DK12)
        term3 = -jnp.einsum('mik, mjd->jdik', solved2, DDK12)
        term4 = -2.0*jnp.einsum('iudk,ip->pudk', jnp.einsum('jdk,jiu->iudk', jnp.einsum('mk, mjd->jdk', solved1, solvedDK11), solvedDK11), DK12)
        term5 = jnp.einsum('mklu,mj->jklu', jnp.einsum('ju,jmkl->mklu', solved1, solvedDDK11), DK12)
        term6 = 2.0*jnp.einsum('jdk,jim->imdk', jnp.einsum('mk, mjd->jdk', solved1, solvedDK11), DDK12)
        term7 = -jnp.einsum('jab,jim->imab', solved2, DDK12)
        term8 = -jnp.einsum('jk,jabc->abck', solved1, DDDK12)
        
        return DDDDK22+term1+term2+term3+term4+term5+term6+term7+term8
    
    def DEmmf(X_test):
        
        J, _ = RM.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
            DJ = RM.DEJ(X_test).reshape(dim, 1,dim)
        else:
            DJ = RM.DEJ(X_test)
            
        DSigma = RM.Dcov(X_test)
  
        termJ = jnp.einsum('ijk,mj->imk', DJ, J)+jnp.einsum('mj,ijk->mik', J, DJ)
        
        return termJ+N_obs*DSigma
    
    def DDEmmf(X_test):
        
        J, Sigma = RM.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
            DJ = RM.DEJ(X_test).reshape(dim, 1,dim)
            DDJ = RM.DDEJ(X_test).reshape(dim, 1, dim, dim)
        else:
            DJ = RM.DEJ(X_test)
            DDJ = RM.DDEJ(X_test)
            
        DDSigma = RM.DDcov(X_test)
        
        termDJ = 2.0*jnp.einsum('ijk,ljd->ildk', DJ, DJ)
        termDDJ = jnp.einsum('ijkl,dj->idkl', DDJ, J)+jnp.einsum('dj, ijkl->idkl', J, DDJ)
        
        G = J.dot(J.T)+N_obs*Sigma
        
        return G, termDJ+termDDJ+N_obs*DDSigma
    
    def DEmmf_inv(X_test):
        
        G = RM.G(X_test)
        G_inv = jnp.linalg.inv(G)
        DG = RM.DG(X_test)
        
        return G_inv, -jnp.einsum('ijk,jm->imk', G_inv.dot(DG), G_inv)
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
        
    dim, N_training = X_training.shape
    
    if y_training.ndim == 1:
        N_obs = 1
    else:
        N_obs = y_training.shape[-1]

    if Dm_fun is None and m_fun is not None:
        Dm_fun = vmap(lambda x: grad(m_fun))
    elif m_fun is None and Dm_fun is None:
        Dm_fun = lambda x: jnp.zeros((x.shape[-1], dim))
        DDm_fun = lambda x: jnp.zeros((x.shape[-1], dim, dim))
        DDDm_fun = lambda x: jnp.zeros((x.shape[-1], dim, dim, dim))
    
    if m_fun is None:
        m_fun = lambda x: jnp.zeros(x.shape[-1])
        
    if k_fun is None:
        k_fun = km.gaussian_kernel
    if Dk_fun is None:
        Dk_fun = grad(k_fun, argnums=0)
    if DDk_fun is None:
        DDk_fun = jacfwd(jacrev(k_fun, argnums=0), argnums=1)
    if DDDk_fun is None:
        DDDk_fun = jacfwd(jacfwd(jacrev(k_fun, argnums=0), argnums=1), argnums=0)
    if DDDDk_fun is None:
        DDDDk_fun = jacfwd(jacfwd(jacfwd(jacrev(k_fun, argnums=0), argnums=1), argnums=0), argnums=1)
        
    DDDk_fun = jacfwd(DDk_fun)
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*delta_stable
    K11_inv = jnp.linalg.inv(K11)
    
    DK11 = vmap(lambda y: vmap(lambda x: Dk_fun(x, y))(X_training.T))(X_training.T)
    DDK11 = vmap(lambda y: vmap(lambda x: DDk_fun(x, y))(X_training.T))(X_training.T)
    
    solvedDK11 = jnp.einsum('jik,im->jmk', DK11, K11_inv)
    solvedDDK11 = jnp.einsum('jikl,im->jmkl', DDK11, K11_inv)
    
    m_training = m_fun(X_training)
    Dm_training = Dm_fun(X_training)
    DDm_training = DDm_fun(X_training)
    
    m_training = m_fun(X_training)
    Dm_training = Dm_fun(X_training)

    RM = riemannian_manifold()
    RM.G = jit(Emmf)
    RM.G_inv = jit(lambda x: jnp.linalg.inv(RM.G(x)))
    RM.DG_inv = jit(DEmmf_inv)
    
    RM.post_mom = jit(post_mom)
    RM.jac_mom = jacobian_mom
    RM.DG = jit(DEmmf)
    RM.DDG = jit(DDEmmf)
    RM.Dcov = jit(Dcov)
    RM.DDcov = jit(DDcov)
    RM.DEJ = jit(DEJ)
    RM.DDEJ = jit(DDEJ)

    RM.chris = jit(chris_symbols)
    RM.Dchris = jit(Dchris)
    RM.geo_ivp = jit(ivp_geodesic)
    RM.geo_bvp = bvp_geodesic
    RM.pt = jit(pt)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    
    return RM

#%% Riemannian manifold with stochastic metric

def RM_SG(X_training, y_training, sigman = 1.0, m_fun = None, Dm_fun = None, k_fun = None, Dk_fun = None, 
          DDk_fun = None, DDDk_fun = None, DDDDk_fun = None,
      theta_init = None, optimize=False, grad_K = None,  max_iter = 100, delta_stable=1e-10,
      grid=jnp.linspace(0,1,100), method='euler', tol=1e-05):
    
    def chris_symbols(x, eps):
                
        G, DG = RM.DG(x, eps)
        G_inv = jnp.linalg.inv(G)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def Dchris(x, eps):
        
        _, DG = RM.DG(x, eps)
        G, DDG = RM.DDG(x, eps)
        G_inv, DG_inv = RM.DG_inv(x, eps)
        
        Dchris1 = jnp.einsum('jlik, lm->mjik', DDG, G_inv) \
                    +jnp.einsum('lijk, lm->mjik', DDG, G_inv) \
                    -jnp.einsum('ijlk, lm->mjik', DDG, G_inv)
        Dchris2 = jnp.einsum('jli, lmk->mjik', DG, DG_inv) \
                    +jnp.einsum('lij, lmk->mjik', DG, DG_inv) \
                    -jnp.einsum('ijl, lmk->mjik', DG, DG_inv)
        
        return 0.5*(Dchris1+Dchris2)
    
    def curvature_operator(x, eps):
        
        Dchris = RM.Dchris(x, eps)
        chris = RM.chris(x, eps)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x, eps):
        
        CO = RM.CO(x, eps)
        G, _ = RM.G(x, eps)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2, eps):
        
        CT = RM.CT(x, eps)[0,1,1,0]
        G, _ = RM.G(x, eps)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v, eps):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, eps)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))

        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y, eps):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, eps)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def pt(v0, gamma, Dgamma, eps):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat, eps)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        fun = jit(eq_pt)
        v = oi.ode_integrator(v0, fun, grid = grid, method=method)
        
        return v
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        N_test = X_test.shape[-1]
        m_test = m_fun(X_test)
        
        K21 = km.km(X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K21).T
        if N_obs == 1:
            mu_post = m_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: m_test+(solved @ (y-m_training)))(y_training)
            
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post+jnp.eye(N_test)*delta_stable
    
    def jacobian_mom(X_test):

        X_test = X_test.reshape(-1)        
        Dm_test = Dm_fun(X_test.reshape(-1,1)).squeeze()    
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK = vmap(lambda x: Dk_fun(x, X_test))(X_training.T)
        DDK = DDk_fun(X_test, X_test)  
        
        solved = jnp.linalg.solve(K11, DK).T
        
        if N_obs == 1:
            mu_post = Dm_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: Dm_test+(solved @ (y-m_training)))(y_training)
        
        cov_post = DDK-(solved @ DK)
        
        return mu_post.T, cov_post+jnp.eye(dim)*delta_stable #SOMETHING WRONG COV IS NOT POSTIVE (SYMMETRIC DEFINITE), BUT IS NEGATIVE SYMMETRIC DEFINITE
    
    def DEJ(X_test):
        
        def DEJ_single(y):
            
            y_diff = y-m_training
            
            term1 = jnp.einsum('ikm,i->km', solved2, y_diff)
            term2 = -jnp.einsum('jk,jl->lk', solved1, jnp.einsum('jmk,m->jk', solvedDK11, y_diff)+Dm_training)

            return DDm_test+term1+term2
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        
        DDm_test = DDm_fun(X_test.reshape(-1,1)).squeeze()
        
        if N_obs == 1:
            return DEJ_single(y_training)
        else:
            return jnp.transpose(vmap(DEJ_single)(y_training), axes=(1,0,2))
        
    def DDEJ(X_test):
        
        def DDEJ_single(y):
            
            y_diff = y-m_training
            
            term1 = jnp.einsum('jikm,j->ikm', solved3, y_diff)
            term2 = -2.0*jnp.einsum('jlik,j->lik', jnp.einsum('mik,mjl->jlik', solved2, solvedDK11), y_diff)
            term3 = -jnp.einsum('mik, mj->jik', solved2, Dm_training)
            term4 = 2.0*jnp.einsum('idlk,i->dlk', jnp.einsum('mlk,mid->idlk', jnp.einsum('jk,jml->mlk', solved1, solvedDK11), solvedDK11), y_diff)
            term5 = -jnp.einsum('mulk,m->ulk', jnp.einsum('jk,jmul->mulk', solved1, solvedDDK11), y_diff)
            term6 = -jnp.einsum('mik,mj->jik', solved2, Dm_training)
            term7 = -jnp.einsum('ik,iml->mlk', solved1, DDm_training)
            
            return DDDm+term1+term2+term3+term4+term5+term6+term7
        
        X_test = X_test.reshape(-1)
        DDDm = DDDm_fun(X_test.reshape(-1,1)).squeeze()
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK12 = vmap(lambda x: DDDk_fun(X_test, x))(X_training.T)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        solved3 = jnp.einsum('jikm, jl->likm', DDDK12, K11_inv)

        if N_obs == 1:
            return DDEJ_single(y_training)
        else:
            return jnp.einsum('ijkm->jikm', vmap(DDEJ_single)(y_training))
    
    def Dcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK = DDDk_fun(X_test, X_test)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        
        term1 = -jnp.einsum('mik,ml->lik', solved2, DK12)
        term2 = jnp.einsum('mdk,ml->ldk', jnp.einsum('jk,jmd->mdk', solved1, solvedDK11), DK12)
        term3 = -jnp.einsum('jk,jmd->mdk', solved1, DDK12)
        
        return DDDK+term1+term2+term3
    
    def DDcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T)
        DDK12 = vmap(lambda x: DDk_fun(X_test, x))(X_training.T)
        DDDK12 = vmap(lambda x: DDDk_fun(X_test, x))(X_training.T)
        DDDDK22 = DDDDk_fun(X_test, X_test)
        
        solved1 = jnp.einsum('ik,ij->jk', DK12, K11_inv)
        solved2 = jnp.einsum('jik, jm->mik', DDK12, K11_inv)
        solved3 = jnp.einsum('jikm, jl->likm', DDDK12, K11_inv)
        
        term1 = -jnp.einsum('likm,ld->dikm', solved3, DK12)
        term2 = 2.0*jnp.einsum('jdik,ju->udik', jnp.einsum('mik, mjd->jdik', solved2, solvedDK11), DK12)
        term3 = -jnp.einsum('mik, mjd->jdik', solved2, DDK12)
        term4 = -2.0*jnp.einsum('iudk,ip->pudk', jnp.einsum('jdk,jiu->iudk', jnp.einsum('mk, mjd->jdk', solved1, solvedDK11), solvedDK11), DK12)
        term5 = jnp.einsum('mklu,mj->jklu', jnp.einsum('ju,jmkl->mklu', solved1, solvedDDK11), DK12)
        term6 = 2.0*jnp.einsum('jdk,jim->imdk', jnp.einsum('mk, mjd->jdk', solved1, solvedDK11), DDK12)
        term7 = -jnp.einsum('jab,jim->imab', solved2, DDK12)
        term8 = -jnp.einsum('jk,jabc->abck', solved1, DDDK12)
        
        return DDDDK22+term1+term2+term3+term4+term5+term6+term7+term8
    
    def mmf(X_test, epsilon):
        
        J, cov = RM.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
        
        L = jnp.linalg.cholesky(cov)
        
        G = J+L.dot(epsilon)
        G = G.dot(G.T)
        
        return G, jnp.linalg.inv(G)
    
    def Dmmf(X_test, epsilon):
        
        J, cov = RM.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
            DJ = RM.DEJ(X_test).reshape(dim, 1,dim)
        else:
            DJ = RM.DEJ(X_test)
            
        DSigma = RM.Dcov(X_test)

        W = epsilon.dot(epsilon.T)

        L = jnp.linalg.cholesky(cov)
        L_inv = jnp.linalg.inv(L)
        
        DL = 0.5*jnp.einsum('ijk, lj->ilk', DSigma, L_inv)
  
        termJ = jnp.einsum('ijk,mj->imk', DJ, J)+jnp.einsum('mj,ijk->mik', J, DJ)
        
        term1 = jnp.einsum('jik,im->jmk', DL, W)
        term2 = jnp.einsum('jmk,im->jik', term1, L)
        term3 = jnp.einsum('ji,im->jm', L, W)
        term4 = jnp.einsum('jm,imk->jik', term3, DL)
        term11 = term2+term4
        
        term5 = jnp.einsum('ji,mi->jm', epsilon, J)
        term6 = jnp.einsum('jik,im->jmk', DL, term5)
        term7 = jnp.einsum('mi,jik->mjk', epsilon, DJ)
        term8 = jnp.einsum('im,mjk->ijk', L, term7)
        term22 = term6+term8
        
        term33 = jnp.transpose(term8, axes=(1,0,2))
        
        G = J+L.dot(epsilon)
        G = G.dot(G.T)
        
        return G, termJ+term11+term22+term33
    
    def DDmmf(X_test, epsilon):
        
        J, cov = RM.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
            DJ = RM.DEJ(X_test).reshape(dim, 1,dim)
            DDJ = RM.DDEJ(X_test).reshape(dim, 1, dim, dim)
        else:
            DJ = RM.DEJ(X_test)
            DDJ = RM.DDEJ(X_test)
            
        DSigma = RM.Dcov(X_test)
        DDSigma = RM.DDcov(X_test)

        W = epsilon.dot(epsilon.T)

        L = jnp.linalg.cholesky(cov)
        L_inv = jnp.linalg.inv(L)
        
        DL = 0.5*jnp.einsum('ijk, lj->ilk', DSigma, L_inv)
        DDL = 2.0*jnp.einsum('ild,jlm->ijdm', DL, DL)
        DDL = 0.5*jnp.einsum('imdk,ji->jmdk', DDSigma-DDL, L_inv)
        
        termDJ = 2.0*jnp.einsum('ijk,ljd->ildk', DJ, DJ)
        termDDJ = jnp.einsum('ijkl,dj->idkl', DDJ, J)+jnp.einsum('dj, ijkl->idkl', J, DDJ)
        
        term1 = jnp.einsum('imkl,jm->ijkl', jnp.einsum('ijkl,jm->imkl', DDL, W), L)
        term2 = jnp.einsum('imkl->mikl', term1)
        term3 = 2.0*jnp.einsum('ijk,ljd->ilkd', jnp.einsum('ilk,lj->ijk', DL, W), DL)
        term11 = term1+term2+term3
        
        term1 = jnp.einsum('imkl,jm->ijkl', jnp.einsum('ijkl,jm->imkl', DDL, epsilon), J)
        term2 = jnp.einsum('im,jmkl->ijkl', jnp.einsum('ij,jm->im', L, epsilon), DDJ)
        term3 = 2.0*jnp.einsum('ijk,ljd->ilkd', jnp.einsum('ilk,lj->ijk', DL, epsilon), DJ)
        term22 = term1+term2+term3
        
        term33 = jnp.einsum('ilkd->likd', term22)
        
        G = J+L.dot(epsilon)
        G = G.dot(G.T)
        
        return G, termDJ+termDDJ+term11+term22+term33
    
    def mmf_inv(X_test, epsilon):
        
        _, G_inv = RM.G(X_test, epsilon)
        
        return G_inv
    
    def Dmmf_inv(X_test, epsilon):
        
        G, G_inv = RM.G(X_test, epsilon)
        _, DG = RM.DG(X_test, epsilon)
        
        return G_inv, -jnp.einsum('ijk,jm->imk', G_inv.dot(DG), G_inv)
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
        
    dim, N_training = X_training.shape
    
    if y_training.ndim == 1:
        N_obs = 1
    else:
        N_obs = y_training.shape[0]
    
    if Dm_fun is None and m_fun is not None:
        Dm_fun = vmap(lambda x: grad(m_fun))
    elif m_fun is None and Dm_fun is None:
        Dm_fun = lambda x: jnp.zeros((x.shape[-1], dim))
        DDm_fun = lambda x: jnp.zeros((x.shape[-1], dim, dim))
        DDDm_fun = lambda x: jnp.zeros((x.shape[-1], dim, dim, dim))
    
    if m_fun is None:
        m_fun = lambda x: jnp.zeros(x.shape[-1])
        
    if k_fun is None:
        k_fun = km.gaussian_kernel
    if Dk_fun is None:
        Dk_fun = grad(k_fun, argnums=0)
    if DDk_fun is None:
        DDk_fun = jacfwd(jacrev(k_fun, argnums=0), argnums=1)
    if DDDk_fun is None:
        DDDk_fun = jacfwd(jacfwd(jacrev(k_fun, argnums=0), argnums=1), argnums=0)
    if DDDDk_fun is None:
        DDDDk_fun = jacfwd(jacfwd(jacfwd(jacrev(k_fun, argnums=0), argnums=1), argnums=0), argnums=1)
        
    DDDk_fun = jacfwd(DDk_fun)
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*delta_stable
    K11_inv = jnp.linalg.inv(K11)
    
    DK11 = vmap(lambda y: vmap(lambda x: Dk_fun(x, y))(X_training.T))(X_training.T)
    DDK11 = vmap(lambda y: vmap(lambda x: DDk_fun(x, y))(X_training.T))(X_training.T)
    
    solvedDK11 = jnp.einsum('jik,im->jmk', DK11, K11_inv)
    solvedDDK11 = jnp.einsum('jikl,im->jmkl', DDK11, K11_inv)
    
    m_training = m_fun(X_training)
    Dm_training = Dm_fun(X_training)
    DDm_training = DDm_fun(X_training)

    RM = riemannian_manifold()
    
    RM.G = jit(mmf)
    RM.DG = jit(Dmmf)
    RM.DDG = DDmmf
    
    RM.G_inv = jit(mmf_inv)
    RM.DG_inv = jit(Dmmf_inv)
    
    RM.post_mom = jit(post_mom)
    RM.jac_mom = jit(jacobian_mom)

    RM.Dcov = jit(Dcov)
    RM.DDcov = jit(DDcov)
    
    RM.DEJ = jit(DEJ)
    RM.DDEJ = jit(DDEJ)
    
    RM.chris = jit(chris_symbols)
    RM.Dchris = jit(Dchris)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    
    RM.geo_ivp = ivp_geodesic
    RM.geo_bvp = bvp_geodesic
    RM.pt = jit(pt)
    
    return RM