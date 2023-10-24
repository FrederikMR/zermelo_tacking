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

#%% Quotient

def horz_vert_split(x:Tuple[ndarray, ndarray],
                    proj:Callable[[Tuple[ndarray, ndarray]], ndarray],
                    sigma:ndarray,
                    G:object,
                    M:object
                    )->Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    
    """ compute kernel of proj derivative with respect to inv A metric """
    
    rank = M.dim
    Xframe = jnp.tensordot(G.invpf(x,G.eiLA),sigma,(2,0))
    Xframe_inv = jnp.linalg.pinv(Xframe.reshape((-1,G.dim)))
    dproj = jnp.einsum('...ij,ijk->...k',jacrev(proj)(x), Xframe)
    (_,_,Vh) = jnp.linalg.svd(jax.lax.stop_gradient(dproj),full_matrices=True)
    ns = Vh[rank:].T # null space
    proj_ns = jnp.tensordot(ns,ns,(1,1))    
    horz = Vh[0:rank].T # horz space
    proj_horz = jnp.tensordot(horz,horz,(1,1))
    
    return (Xframe,Xframe_inv,proj_horz,proj_ns,horz)

def get_sde_fiber(sde_f:Callable[[ndarray,ndarray], ndarray],
                  proj:Callable[[ndarray], ndarray],
                  G:object,
                  M:object
                  ):
    
    """hit target v at time t=Tend"""
    
    def sde_fiber(c:Tuple[ndarray, ndarray, ndarray, ndarray],
                  y:Tuple[ndarray, ndarray]
                  ):
        (det,sto,X,*dys_sde) = sde_f(c,y)
        t,g,_,sigma = c
        dt,dW = y
        
        (Xframe,Xframe_inv,_,proj_ns,_) = horz_vert_split(g,proj,sigma,G,M)
        
        det = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,X.reshape((-1,G.dim)),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_fiber

def get_sde_horz(sde_f:Callable[[ndarray, ndarray], ndarray],
                 proj:Callable[[ndarray], ndarray],
                 G:object,
                 M:object
                 ):
    
    def sde_horz(c:Tuple[ndarray, ndarray, ndarray, ndarray],
                 y:Tuple[ndarray, ndarray]
                 )->Tuple[ndarray, ndarray, ndarray,...]:
        
        (det,sto,X,*dys_sde) = sde_f(c,y)
        t,g,_,sigma = c
        dt,dW = y
        
        (Xframe,Xframe_inv,proj_horz,_,_) = horz_vert_split(g,proj,sigma,G,M)        
        det = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,X.reshape((-1,G.dim)),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_horz

def get_sde_lifted(sde_f:Callable[[ndarray, ndarray], ndarray],
                   proj:Callable[[ndarray], ndarray],
                   G:object,
                   M:object
                   ):
                              
    def sde_lifted(c:Tuple[ndarray, ndarray, ndarray, ndarray],
                   y:Tuple[ndarray, ndarray]
                   ):
        
        t,g,chart,sigma,*cs = c
        dt,dW = y

        (det,sto,X,*dys_sde) = sde_f((t,M.invF((proj(g),chart)),chart,*cs),y)
        
        (Xframe,Xframe_inv,proj_horz,_,horz) = horz_vert_split(g,proj,sigma,G,M) 

        
        det = jnp.tensordot(Xframe,jnp.tensordot(horz,det,(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(horz,sto,(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(horz,X,(1,0)),(2,0)).reshape((G.dim,G.dim,M.dim))
        
        return (det,sto,X,jnp.zeros_like(sigma),*dys_sde)

    return sde_lifted