    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J poincarre_stochastic
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
    
    python3 tacking.py \
        --manifold poincarre \
        --geometry stochastic \
        --method adam \
        --T 1000 \
        --lr_rate 0.01 \
        --tol 0.0001 \
        --max_iter 1000 \
        --sub_iter 5 \
        --N_sim 5 \
        --seed 2712 \
        --save_path tacking_gpu/ \
    