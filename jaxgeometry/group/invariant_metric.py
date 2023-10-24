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

#%% Invariant Metric

def initialize(G:object,
               _sigma:ndarray=None
               )->None:
    """ add left-/right-invariant metric related structures to group 

    parameter sigma is square root cometric / diffusion field
    """

    def gV(v:ndarray=None,
           w:ndarray=None,
           sigma:ndarray=_sigma
           )->ndarray:
        
        if v is None and w is None:
            return G.A(sigma)
        elif v is not None and w is None:
            return jnp.tensordot(G.A(sigma),v,(1,0))
        elif v.ndim == 1 and w.ndim == 1:
            return jnp.dot(v,jnp.dot(G.A(sigma),w))
        elif v.ndim == 1 and not w:
            return jnp.dot(G.A(sigma),v)
        elif v.ndim == 2 and w.ndim == 2:
            return jnp.tensordot(v,jnp.tensordot(G.A(sigma),w,(1,0)),(0,0))
        else:
            assert(False)
            
    def cogV(cov:ndarray=None,
             cow:ndarray=None,
             sigma:ndarray=_sigma
             )->ndarray:
        
        if cov is None and cow is None:
            return G.W(sigma)
        elif cov is not None and cow is None:
            return jnp.tensordot(G.W(sigma),cov,(1,0))
        elif cov.ndim == 1 and cow.ndim == 1:
            return jnp.dot(cov,jnp.dot(G.W(sigma),cow))
        elif cov.ndim == 2 and cow.ndim == 2:
            return jnp.tensordot(cov,jnp.tensordot(G.W(sigma),cow,(1,0)),(0,0))
        else:
            assert(False)
            
    def gLA(xiv:ndarray,
            xiw:ndarray,
            sigma:ndarray=_sigma
            )->ndarray:
        
        v = G.LAtoV(xiv)
        w = G.LAtoV(xiw)
        
        return G.gV(v,w,sigma)
    
    def cogLA(coxiv:ndarray,
              coxiw:ndarray,
              sigma:ndarray=_sigma
              )->ndarray:
        
        cov = G.LAtoV(coxiv)
        cow = G.LAtoV(coxiw)
        
        return G.cogV(cov,cow,sigma)
    
    def gG(g:ndarray,
           vg:ndarray,
           wg:ndarray,
           sigma:ndarray=_sigma
           )->ndarray:
        
        xiv = G.invpb(g,vg)
        xiw = G.invpb(g,wg)
        
        return G.gLA(xiv,xiw,sigma)
    
    def gpsi(hatxi:ndarray,
             v:ndarray=None,
             w:ndarray=None,
             sigma:ndarray=_sigma
             )->ndarray:
        
        g = G.psi(hatxi)
        vg = G.dpsi(hatxi,v)
        wg = G.dpsi(hatxi,w)
        
        return G.gG(g,vg,wg,sigma)
    
    def cogpsi(hatxi:ndarray,
               p:ndarray=None,
               pp:ndarray=None,
               sigma:ndarray=_sigma
               )->ndarray:
        
        invgpsi = G.inv(G.gpsi(hatxi,sigma=sigma))
        if p is not None and pp is not None:
            return jnp.tensordot(p,jnp.tensordot(invgpsi,pp,(1,0)),(0,0))
        elif p and not pp:
            return jnp.tensordot(invgpsi,p,(1,0))
        return invgpsi

    # sharp/flat mappings
    def sharpV(mu:ndarray,
               sigma:ndarray=_sigma
               )->ndarray:
        
        return jnp.dot(G.W(sigma),mu)
    
    def flatV(v:ndarray,
              sigma:ndarray=_sigma
              )->ndarray:
        
        return jnp.dot(G.A(sigma),v)
    
    def sharp(g:ndarray,
              pg,sigma:ndarray=_sigma
              )->ndarray:
        
        return G.invpf(g,G.VtoLA(jnp.dot(G.W(sigma),G.LAtoV(G.invcopb(g,pg)))))
    
    def flat(g:ndarray,
             vg:ndarray,
             sigma:ndarray=_sigma
             )->ndarray:
        
        return G.invcopf(g,G.VtoLA(jnp.dot(G.A(sigma),G.LAtoV(G.invpb(g,vg)))))
    
    def sharppsi(hatxi:ndarray,
                 p:ndarray,
                 sigma:ndarray=_sigma
                 )->ndarray:
        
        return jnp.tensordot(G.cogpsi(hatxi,sigma=sigma),p,(1,0))
    
    def flatpsi(hatxi:ndarray,
                v:ndarray,
                sigma:ndarray=_sigma
                )->ndarray:
        
        return jnp.tensordot(G.gpsi(hatxi,sigma=sigma),v,(1,0))
    
    if _sigma is None:
        _sigma = jnp.eye(G.dim)

    G.sqrtA = lambda sigma=_sigma: G.inv(sigma) # square root metric
    G.A = lambda sigma=_sigma: jnp.tensordot(G.sqrtA(sigma),G.sqrtA(sigma),(0,0)) # metric
    G.W = lambda sigma=_sigma: jnp.tensordot(sigma,sigma,(1,1)) # covariance (cometric)
    
    G.gV = gV
    G.cogV = cogV
    G.gLA = gLA
    G.cogLA = cogLA
    G.gG = gG
    G.gpsi = gpsi
    G.cogpsi = cogpsi
    G.sharpV = sharpV
    G.flatV = flatV
    G.sharp = sharp
    G.flat = flat
    G.sharppsi = sharppsi
    
    G.flatpsi = flatpsi
    
    return