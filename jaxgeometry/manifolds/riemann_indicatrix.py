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
import jaxgeometry.manifolds.riemannian as riemannian

#%% Euclidean Geometry (R^n)

class RiemannIndicatrix(riemannian.Manifold):
    """ 2d Indicatrix """

    def __init__(self,
                 mua_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.ones(1, dtype=jnp.float32),
                 mub_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.ones(1, dtype=jnp.float32),
                 mutheta_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.zeros(1, dtype=jnp.float32),
                 sigmaa_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.zeros(1, dtype=jnp.float32),
                 sigmab_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.zeros(1, dtype=jnp.float32),
                 sigmatheta_fun:Callable[[Tuple[ndarray, ndarray]], ndarray] = lambda x: jnp.zeros(1, dtype=jnp.float32),
                 eps:ndarray = jnp.zeros(3)
                 )->None:
        riemannian.Manifold.__init__(self)
        self.dim = 2

        self.do_chart_update = lambda x: False
        self.update_coords = lambda coords,_: coords

        ##### Metric:
        def G(x):
    
            theta = mutheta_fun(x)+eps[0]*sigmatheta_fun(x)
            a2 = (mua_fun(x)+eps[1]*sigmaa_fun(x))**2
            b2 = (mub_fun(x)+eps[2]*sigmab_fun(x))**2
            costheta = jnp.cos(theta)
            sintheta = jnp.sin(theta)
            
            return jnp.array([[a2*sintheta**2+b2*costheta**2, (a2-b2)*costheta*sintheta],
                             [(a2-b2)*costheta*sintheta, a2*costheta**2+b2*sintheta**2]]).squeeze()/(a2*b2)
        self.g = lambda x: G(x)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        riemannian.metric(self)
        riemannian.curvature(self)
        riemannian.geodesic(self)
        riemannian.Log(self)
        riemannian.parallel_transport(self)
        
        #Metric
        #self.Gamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        #self.DGamma_g = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim, self.dim)))
        #self.gsharp = jit(lambda x: jnp.eye(self.dim))
        #self.Dg = jit(lambda x: jnp.zeros((self.dim, self.dim, self.dim)))
        #self.mu_Q = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.det = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.detsharp = jit(lambda x: jnp.ones(1, dtype=jnp.float32))
        #self.logAbsDet = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        #self.logAbsDetsharp = jit(lambda x: jnp.zeros(1, dtype=jnp.float32))
        #self.dot = jit(lambda x,v,w: v.dot(w))
        #self.dotsharp = jit(lambda x, p, pp: pp.dot(p))
        #self.flat = jit(lambda x,v: v)
        #self.sharp = jit(lambda x,p: p)
        #self.orthFrame = jit(lambda x: jnp.eye(self.dim))
        #self.div = lambda x,X: jnp.trace(jacfwdx(X)(x))
        #self.divsharp = lambda x,X: jnp.trace(jacfwdx(X)(x))
        
        #Geodesic
        #self.geodesic = jit(lambda x,v,dts: (jnp.cumsum(dts), jnp.stack((x[0]+jnp.cumsum(dts)[:,None]*v, 
        #                                                                 jnp.tile(v, (len(dts), 1)))).transpose(1,0,2), 
        #                                     jnp.tile(x[1], (len(dts), 1))))
        
        #Log
        #self.Log = jit(lambda x,y: y[0]-x[0])
        #self.dist = jit(lambda x,y: jnp.sqrt(jnp.sum((x[0]-y[0])**2)))
        
        #Parallel Transport - ADD CLOSED FORM EXPRESSIONS
        
        #Curvature - ADD CLOSED FORM EXPRESSIONS
        
        return
    
    def update_vector(self, coords:ndarray, new_coords:ndarray, new_chart:ndarray, v:ndarray)->ndarray:
        
        return v

    def __str__(self)->str:
        
        return "Riemannian manifold based on indatrix-fields of dimension %d" % (self.dim)

    def plot(self)->None:
        if self.dim == 2:
            plt.axis('equal')
    
    def plot_path(self, xs:Tuple[ndarray, ndarray], 
                  u:ndarray=None, 
                  color:str='b', 
                  color_intensity:float=1., 
                  linewidth:float=1., 
                  prevx:Tuple[ndarray, ndarray]=None, 
                  last:bool=True, 
                  s:int=20, 
                  arrowcolor:str='k'
                  )->None:
        
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==N-1)
            prevx = x
            
        return

    def plotx(self, x:Tuple[ndarray, ndarray], 
              u:ndarray=None, 
              color:str='b', 
              color_intensity:float=1., 
              linewidth:float=1., 
              prevx:Tuple[ndarray, ndarray]=None,
              last:bool=True, 
              s:int=20, 
              arrowcolor:str='k'
              )->None:
        assert(type(x) == type(()) or x.shape[0] == self.dim)
        if type(x) == type(()):
            x = x[0]
        if type(prevx) == type(()):
            prevx = prevx[0]

        ax = plt.gca()

        if last:
            if self.dim == 2:
                plt.scatter(x[0],x[1],color=color,s=s)
            elif self.dim == 3:
                ax.scatter(x[0],x[1],x[2],color=color,s=s)
        else:
            try:
                xx = np.stack((prevx,x))
                if self.dim == 2:
                    plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
                elif self.dim == 3:
                    ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
            except:
                if self.dim == 2:
                    plt.scatter(x[0],x[1],color=color,s=s)
                elif self.dim == 3:
                    ax.scatter(x[0],x[1],x[2],color=color,s=s)

        try:
            plt.quiver(x[0], x[1], u[0], u[1], pivot='tail', linewidth=linewidth, scale=5, color=arrowcolor)
        except:
            pass
        
        return





















