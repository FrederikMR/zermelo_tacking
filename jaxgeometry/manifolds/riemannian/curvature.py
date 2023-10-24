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

#https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry

#%% Modules

from jaxgeometry.setup import *

#%% Riemannian curvature

def initialize(M:object) -> None:
    
    """ Riemannian curvature """
    
    @jit
    def CurvatureOperator(x: Tuple[ndarray, ndarray]) -> ndarray:
        
        """
        Riemannian Curvature tensor
        
        Args:
            x: point on manifold
        
        Returns:
            4-tensor R_ijk^l in with order i,j,k,l
            (see e.g. https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#(3,1)_Riemann_curvature_tensor )
            Note that sign convention follows e.g. Lee, Riemannian Manifolds.
        """
        
        chris = M.Gamma_g(x)
        Dchris = M.DGamma_g(x)
        
        return -(jnp.einsum('pik,ljp->ijkl',chris,chris)
                    -jnp.einsum('pjk,lip->ijkl',chris,chris)
                    +jnp.einsum('likj->ijkl',Dchris)
                    -jnp.einsum('ljki->ijkl',Dchris))
    
    @jit
    def CurvatureForm(x: Tuple[ndarray, ndarray])->ndarray:
        
        """
        Riemannian Curvature form
        R_u (also denoted Omega) is the gl(n)-valued curvature form u^{-1}Ru for a frame
        u for T_xM
        
        Args:
            x: point on manifold
        
        Returns:
            4-tensor (R_u)_ij^m_k with order i,j,m,k
        """
        M.R_u = jit(lambda x,u: jnp.einsum('ml,ijql,qk->ijmk',jnp.linalg.inv(u),M.R(x),u))
        
    @jit
    def CurvatureTensor(x: Tuple[ndarray, ndarray]) -> ndarray:
        
        return jnp.einsum('sijk,sm->ijkm', M.R(x), M.g(x))
        
    @jit
    def RicciCurvature(x: Tuple[ndarray, ndarray]) -> ndarray:
        
        """
        Ricci curvature
        
        Args:
            x: point on manifold
        
        Returns:
            2-tensor R_ij in order i,j
        """
        
        return jnp.einsum('kijk->ij', M.R(x))
    
    @jit
    def TracelessRicci(x:Tuple[ndarray, ndarray])->ndarray:
        
        G = M.g(x)
        R = M.Ricci_curv(x)
        S = M.S_curv(x)
        
        return R-S*G/M.dim
    
    @jit
    def EinsteinTensor(x: Tuple[ndarray, ndarray]) -> ndarray:
        
        R = M.Ricci_curv(x)
        S = M.S_curv(x)
        G = M.g(x)
        
        return R-0.5*S*G
    
    @jit
    def ScalarCurvature(x: Tuple[ndarray, ndarray]) -> ndarray:
        
        """
        Scalar curvature
        
        Args:
            x: point on manifold
        
        Returns:
            scalar curvature
        """
        
        return jnp.einsum('ij,ij->', M.gsharp(x),M.Ricci_curv(x))
    
    @jit
    def SectionalCurvature(x: Tuple[ndarray, ndarray], e1:ndarray, e2:ndarray) -> ndarray:
        
        """
            Sectional curvature
            
            Args:
                x: point on manifold
                e1,e2: two orthonormal vectors spanning the section
            
            Returns:
                sectional curvature K(e1,e2)
            """
        
        G = M.g(x)
        CO = M.R(x)
        
        CT = jnp.einsum('sijk,sm->ijkm', CO, G)[0,1,1,0]
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    M.R = CurvatureOperator
    M.R_u = CurvatureForm
    M.Ricci_curv = RicciCurvature
    M.S_curv = ScalarCurvature
    
    M.SectionalCurvature = SectionalCurvature
    M.EinsteinTensor = EinsteinTensor
    M.TracelessRicci = TracelessRicci
    M.CurvatureTensor = CurvatureTensor
    
    
    return