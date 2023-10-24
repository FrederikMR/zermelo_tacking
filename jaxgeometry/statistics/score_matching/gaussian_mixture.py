#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:38:42 2023

@author: fmry
"""

#%% Sources

#https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

#%% Modules

from jaxgeometry.setup import *
import numpy as np

#%% Code

#def euclidean(point, data):
#    """
#    Euclidean distance between point & data.
#    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
#    """
#    return jnp.sqrt(jnp.sum((point - data)**2, axis=1))
class GaussianMixture:
    def __init__(self, p_fun, diffusion_fun, grady_log, gradt_log, n_clusters=4, max_iter=100):
        self.p_fun = p_fun
        self.diffusion_fun = diffusion_fun
        self.grady_log = grady_log
        self.gradt_log = gradt_log
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.key = random.PRNGKey(2712)
    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        key, subkey = random.split(self.key)
        self.key = subkey
        centroid_idx = [random.choice(subkey, jnp.arange(0,len(X_train[0]), 1))]
        self.pi = jnp.ones(self.n_clusters)/self.n_clusters
        self.centroids = (X_train[0][jnp.array(centroid_idx)].reshape(1,-1), 
                          X_train[1][jnp.array(centroid_idx)].reshape(1,-1))
        self.diffusion_time = jnp.ones(self.n_clusters)*0.5
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = vmap(lambda x,chartx,t: vmap(lambda y,charty: \
                                               self.p_fun((x,chartx), 
                                                             (y,charty), t))(X_train[0], 
                                                                          X_train[1]))(self.centroids[0], 
                                                                                       self.centroids[1],
                                                                                       self.diffusion_time[:(_+1)])
            dists = jnp.nan_to_num(jnp.sum(dists, axis=0), nan=0)         
            # Normalize the distances
            dists /= jnp.sum(dists)
            # Choose remaining points based on their distances
            key, subkey = random.split(self.key)
            self.key = subkey
            new_centroid_idx = [random.choice(key, jnp.arange(0,len(X_train[0])), p=dists)]
            centroid_idx += new_centroid_idx
            self.centroids = (X_train[0][jnp.array(centroid_idx)], 
                              X_train[1][jnp.array(centroid_idx)])
            
        for _ in range(self.max_iter):
            # Sort each datapoint, assigning to nearest centroid
            print(f"Iteration {_+1}/{self.max_iter}")
            print(self.centroids[1].shape)
            print(self.diffusion_time.shape)
            gamma_zk = vmap(lambda mu,chartmu,t: \
                            vmap(lambda x,chartx: \
                                 self.p_fun((x,chartx),(mu,chartmu),t))(X_train[0],
                                                                        X_train[1]))(self.centroids[0],
                                                                                     self.centroids[1],
                                                                                     self.diffusion_time)
            gamma_zk = gamma_zk/jnp.sum(gamma_zk, axis=0)
            centroid_idx = jnp.argmax(gamma_zk, axis=0)
            centroid_idx = jnp.stack([centroid_idx==k for k in range(self.n_clusters)])
            prev_centroids = self.centroids
            prev_diffusion = self.diffusion_time
            mu1 = []
            mu2 = []
            diffusion_time = []
            for i in range(self.n_clusters):
                mu, t = self.diffusion_fun((X_train[0],X_train[1]),
                                        (self.centroids[0][i], self.centroids[1][i]),
                                        self.diffusion_time[i],
                                        grady_log=lambda x,y,t: grady_log(x,y,t)/gamma_zk[i],
                                        grady_log=lambda x,y,t: grady_log(x,y,t)/gamma_zk[i])
                mu1.append(mu[0])
                mu2.append(mu[1])
                diffusion_time.append(t)
            self.centroids = (jnp.stack(mu1), jnp.stack(mu2))
            self.diffusion_time = jnp.stack(diffusion_time)
                
                
            for i in range(self.n_clusters):
                if jnp.isnan(self.centroids[0][i]).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids = list(self.centroids)
                    self.centroids[0] = self.centroids[0].at[i].set(prev_centroids[0][i])
                    self.centroids[1] = self.centroids[1].at[i].set(prev_centroids[1][i])
                    self.centroids = tuple(self.centroids)
                    self.diffusion_time = prev_diffusion
                    
        self.centroid_idx = centroid_idx