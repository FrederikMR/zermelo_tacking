#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(manifold, geometry):

    with open ('submit_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J {manifold}_{geometry}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 tacking.py \\
        --manifold {manifold} \\
        --geometry {geometry} \\
        --method adam \\
        --T 1000 \\
        --lr_rate 0.01 \\
        --tol 0.0001 \\
        --max_iter 1000 \\
        --sub_iter 5 \\
        --N_sim 5 \\
        --seed 2712 \\
        --save_path tacking_gpu/ \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    geometries = ['fixed', 'stochastic']
    manifolds = ['direction_only', 'time_only', 'poincarre']
    
    for geo in geometries:
        for man in manifolds:
            time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
            generate_job(man, geo)
            try:
                submit_job()
            except:
                time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                try:
                    submit_job()
                except:
                    print(f"Job script with {man}, {geo} failed!")

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)