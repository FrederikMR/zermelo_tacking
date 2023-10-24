## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Jax Geometry. If not, see <http://www.gnu.org/licenses/>.
#

#%% Sources

#%% Modules

from jaxgeometry.setup import *
from jaxgeometry.statistics.iterative_mle import *
from jaxgeometry.stochastics import get_guided

#%% Code

def initialize(M:object)->None:

    # function to update charts for position depends parameters
    def params_update(state, chart):
        try:
            ((x,m,v),),*s = state
            if M.do_chart_update((x,chart)):
                new_chart = M.centered_chart((x,chart))
                (x,chart) = M.update_coords((x,chart),new_chart)
            return optimizers.OptimizerState(((x,m,v),),*s),chart
        except ValueError: # state is packed
            states_flat, tree_def, subtree_defs = state
            ((x,m,v),*s) = states_flat
            if M.do_chart_update((x,chart)):
                new_chart = M.centered_chart((x,chart))
                (x,chart) = M.update_coords((x,chart),new_chart)
            states_flat = ((x,m,v),*s)
            return (states_flat,tree_def,subtree_defs),chart
        
    if hasattr(M, 'F'):
        F_fun = M.F
    else:
        F_fun = lambda x: x[0]
        
    # guide function
    #phi = lambda q,v,s: jnp.tensordot((1/s)*jnp.linalg.cholesky(M.g(q)).T,M.StdLog(q,M.F((v,q[1]))).flatten(),(1,0))
    #A = lambda x,v,w,s: (s**(-2))*jnp.dot(v,jnp.dot(M.g(x),w))
    #logdetA = lambda x,s: jnp.linalg.slogdet(s**(-2)*M.g(x))[1]
    
    #(Brownian_coords_guided,sde_Brownian_coords_guided,chart_update_Brownian_coords_guided,log_p_T,neg_log_p_Ts) = get_guided(
    #    M,M.sde_Brownian_coords,M.chart_update_Brownian_coords,phi,
    #    lambda x,s: s*jnp.linalg.cholesky(M.gsharp(x)),A,logdetA)
    
    # guide function
    phi = lambda q,v,s: jnp.tensordot((1/s)*jnp.linalg.cholesky(M.g(q)).T,
                                      M.Log(q,F_fun((v,q[1]))).flatten(),
                                      (1,0))
    A = lambda x,v,w,s: (s**(-2))*jnp.dot(v,jnp.dot(M.g(x),w))
    logdetA = lambda x,s: jnp.linalg.slogdet(s**(-2)*M.g(x))[1]
    
    (Brownian_coords_guided,sde_Brownian_coords_guided,chart_update_Brownian_coords_guided,log_p_T,neg_log_p_Ts) = get_guided(
        M,M.sde_Brownian_coords,M.chart_update_Brownian_coords,phi,
        lambda x,s: s*jnp.linalg.cholesky(M.gsharp(x)),A,logdetA)

    # optimization setup
    N = 1 # bridge samples per datapoint
    _dts = dts(n_steps=100,T=1.)

    # define parameters
    x = M.coords(jnp.zeros(M.dim))
    params_inds = (0,5)
    
    M.diffusion_mean = lambda samples,params=(x[0]+.1*np.random.normal(size=M.dim),jnp.array(.2,dtype="float32")),N=N,num_steps=80: \
            iterative_mle(samples,\
                neg_log_p_Ts,\
                params,params_inds,params_update,x[1],_dts,M,\
                N=N,num_steps=num_steps,step_size=1e-2)

    M.diffusion_mean_ll = lambda x, t, obss, N=N: \
        jnp.mean(vmap(lambda w,dW: log_p_T(x,
                                           w,
                                           dW,
                                           _dts,
                                           *t))(obss, 
                                               dWs(len(obss[0])*N*M.dim,_dts).reshape(-1,
                                                                                      _dts.shape[0],
                                                                                      N,
                                                                                      M.dim)))
                                                
    M.transition_density = lambda x0, xt, t, N=N: log_p_T(x0,xt,dWs(N*M.dim,_dts).reshape(_dts.shape[0],
                                                                                         N,
                                                                                         M.dim),_dts,*t)
                                               
    return
    
    
    
    
    
    
    