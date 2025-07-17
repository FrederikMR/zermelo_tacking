    #! /bin/bash
    #BSUB -q hpc
    #BSUB -J time_position_fixed
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "select[model=XeonGold6226R]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 

    module swap python3/3.10.12
    
    python3 tacking.py \
        --manifold time_position \
        --geometry fixed \
        --method adam \
        --T 1000 \
        --lr_rate 0.01 \
        --alpha 1.0 \
        --tol 0.0001 \
        --max_iter 10000 \
        --sub_iter 10 \
        --N_sim 5 \
        --idx_birds 0 \
        --idx_data 0 \
        --seed 2712 \
        --albatross_file_path /work3/fmry/Data/albatross/tracking_data.xls \
        --save_path tacking_cpu/ \
    