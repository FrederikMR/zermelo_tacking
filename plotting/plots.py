#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:30:13 2025

@author: fmry
"""

#%% Modules

from jax import Array

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from typing import List

from abc import ABC

#%%

plt.rcParams.update({'font.size': 25})
cbformat = tkr.ScalarFormatter()   # create the formatter
cbformat.set_powerlimits((-2,2)) 

#%% Plotting

class TackPlots(ABC):
    def __init__(self,
                 font_size:int=25,
                 power_lim:float=-2.,
                 colors:List=['red', 'blue'],
                 linewidth:float = 2.5,
                 s:float = 500,
                 alpha:float = 1.0,
                 )->None:
        
        self.colors = colors
        self.font_size = font_size
        self.power_lim = power_lim
        
        self.s = s
        self.alpha = alpha
        self.linewidth = linewidth
        
        return
    
    def plot_stochastic_tacking(self, 
                                z0:Array,
                                zT:Array,
                                T:int,
                                expected_zs:Array,
                                expected_zs_reverse:Array,
                                expected_tack_curve:Array,
                                expected_reverse_tack_curve:Array,
                                stochastic_zs:Array,
                                stochastic_zs_reverse:Array,
                                stochastic_tack_curve:Array,
                                stochastic_reverse_tack_curve:Array,
                                indicatrix_alpha:Array=None,
                                indicatrix_beta:Array=None,
                                xscales:List=None,
                                yscales:List=None,
                                equal_frame:bool=False,
                                save_path='tacking.pdf',
                                )->None:
        
        plt.rcParams.update({'font.size': self.font_size})
        cbformat = tkr.ScalarFormatter()   # create the formatter
        cbformat.set_powerlimits((-self.power_lim,self.power_lim)) 
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(121)
        
        c1, c2 = self.colors
        
        
        #Curves with no tacking
        if expected_zs is not None:
            ax.plot(expected_zs[:,0], expected_zs[:,1], color=c1, linewidth=self.linewidth, alpha=self.alpha)
        if expected_zs_reverse is not None:
            ax.plot(expected_zs_reverse[:,0], expected_zs_reverse[:,1], color=c2, linewidth=self.linewidth, alpha=self.alpha)
        
        #Plotting tack curves
        for j in range(2):
            
            if j % 2 == 0:
                #Plotting Curves
                ax.plot(expected_tack_curve[(T*j):(T*(j+1))][:,0], 
                        expected_tack_curve[(T*j):(T*(j+1))][:,1], 
                        color=c1,
                        linewidth=self.linewidth)
                
                ax.scatter(expected_tack_curve[(T*(j+1))][0], 
                           expected_tack_curve[(T*(j+1))][1],
                           color=c1,
                           s=self.s)
                    
                    
                ax.plot(expected_reverse_tack_curve[(T*j):(T*(j+1))][:,0], 
                        expected_reverse_tack_curve[(T*j):(T*(j+1))][:,1], 
                        color=c2,
                        linewidth=self.linewidth)
                
                ax.scatter(expected_reverse_tack_curve[(T*(j+1))][0], 
                           expected_reverse_tack_curve[(T*(j+1))][1],
                           color=c2,
                           s=self.s)
            else:
                #Plotting Curves
                ax.plot(expected_tack_curve[(T*j):(T*(j+1))][:,0], 
                        expected_tack_curve[(T*j):(T*(j+1))][:,1], 
                        color=c2,
                        linewidth=self.linewidth)
                
                ax.plot(expected_reverse_tack_curve[(T*j):(T*(j+1))][:,0], 
                        expected_reverse_tack_curve[(T*j):(T*(j+1))][:,1], 
                        color=c1,
                        linewidth=self.linewidth)
                    
        #Indicatrices
        if indicatrix_alpha is not None:
            ax.plot(indicatrix_alpha[:,0]+z0[0], 
                    indicatrix_alpha[:,1]+z0[1], 
                    color=c1, 
                    linestyle='dashed',
                    )
        
        #Indicatrices
        if indicatrix_beta is not None:
            ax.plot(indicatrix_beta[:,0]+zT[0], 
                    indicatrix_beta[:,1]+zT[1], 
                    color=c2, 
                    linestyle='dashed',
                    )
        
        ax.set_xlabel(r'$x^{1}$', fontsize=25)
        ax.set_ylabel(r'$x^{2}$', fontsize=25)
        ax.set_title("Expected Tack Curve", fontsize=25)
        ax.grid(True)
        #ax.set_xlim(xscales[0], xscales[1])
        #ax.set_ylim(yscales[0], yscales[1])
        #ax.set_aspect('equal', adjustable='box')
        
        ax.plot([0],[0], color=c1, label=r'$F^{\alpha}$', linewidth=self.linewidth)
        ax.plot([0],[0], color=c2, label=r'$F^{\beta}$', linewidth=self.linewidth)
        
        #Start and end point
        ax.scatter(z0[0], z0[1], marker="s", color=c1, s=self.s)
        ax.scatter(zT[0], zT[1], marker="s", color=c2, s=self.s)
        
        ax = fig.add_subplot(122)
        
        for szs, szs_reverse, stack_curve, sreverse_tack_curve in zip(stochastic_zs,
                                                                      stochastic_zs_reverse,
                                                                      stochastic_tack_curve, 
                                                                      stochastic_reverse_tack_curve,
                                                                      ):
            
            #Curves with no tacking
            ax.plot(szs[:,0], szs[:,1], color=c1, linewidth=self.linewidth, alpha=0.2)
            ax.plot(szs_reverse[:,0], szs_reverse[:,1], color=c2, linewidth=self.linewidth, alpha=0.2)
            
            #Plotting tack curves
            for j in range(2):
                if j % 2 == 0:
                    #Plotting Curves
                    ax.plot(stack_curve[(T*j):(T*(j+1))][:,0], 
                            stack_curve[(T*j):(T*(j+1))][:,1], 
                            color=c1,
                            linewidth=self.linewidth,
                            alpha=0.2,
                            )
                    ax.plot(sreverse_tack_curve[(T*j):(T*(j+1))][:,0], 
                            sreverse_tack_curve[(T*j):(T*(j+1))][:,1], 
                            color=c2,
                            linewidth=self.linewidth,
                            alpha=0.2,
                            )
        
                    #Plotting points
                    ax.scatter(stack_curve[(T*(j+1))][0], 
                               stack_curve[(T*(j+1))][1],
                               color=c1,
                               s=self.s)
                    ax.scatter(sreverse_tack_curve[(T*(j+1))][0], 
                               sreverse_tack_curve[(T*(j+1))][1],
                               color=c2,
                               s=self.s)
                else:
                    #Plotting Curves
                    ax.plot(stack_curve[(T*j):(T*(j+1))][:,0], 
                            stack_curve[(T*j):(T*(j+1))][:,1], 
                            color=c2,
                            linewidth=self.linewidth,
                            alpha=0.2,
                            )
                    ax.plot(sreverse_tack_curve[(T*j):(T*(j+1))][:,0], 
                            sreverse_tack_curve[(T*j):(T*(j+1))][:,1], 
                            color=c1,
                            linewidth=self.linewidth,
                            alpha=0.2,
                            )
        
        #Indicatrices
        if indicatrix_alpha is not None:
            ax.plot(indicatrix_alpha[:,0]+z0[0], 
                    indicatrix_alpha[:,1]+z0[1], 
                    color=c1, 
                    linestyle='dashed',
                    )
        
        #Indicatrices
        if indicatrix_beta is not None:
            ax.plot(indicatrix_beta[:,0]+zT[0], 
                    indicatrix_beta[:,1]+zT[1], 
                    color=c2, 
                    linestyle='dashed',
                    )
        
        ax.set_xlabel(r'$x^{1}$', fontsize=self.font_size)
        ax.set_ylabel(r'$x^{2}$', fontsize=self.font_size)
        ax.set_title("Stochastic Tack Curve", fontsize=self.font_size)
        ax.grid(True)
        if xscales is not None:
            ax.set_xlim(xscales[0], xscales[1])
        if yscales is not None:
            ax.set_ylim(yscales[0], yscales[1])
        if equal_frame:
            ax.set_aspect('equal', adjustable='box')
        
        ax.plot([0],[0], color=c1, label=r'$F^{\alpha}$', linewidth=self.linewidth)
        ax.plot([0],[0], color=c2, label=r'$F^{\beta}$', linewidth=self.linewidth)
        
        #Start and end point
        ax.scatter(z0[0], z0[1], marker="s", color=c1, s=self.s/2)
        ax.scatter(zT[0], zT[1], marker="s", color=c2, s=self.s/2)
        
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc=(0.17,0.7), ncol=2, fontsize=25)
        
        fig.tight_layout()
        
        fig.savefig(save_path, format='pdf', pad_inches=0.1, bbox_inches='tight')
        
        plt.show()
        
        return fig