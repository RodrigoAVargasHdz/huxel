#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=12  # Cores proportional to GPU 
#SBATCH --mem=64G 
#SBATCH --qos nopreemption  
#SBATCH --partition=cpu 
#SBATCH --job-name=huxel_n_10_l_0_exp_AdamW 
#SBATCH --time=24:00:00 
#SBATCH --output=out_huxel_n_10_l_0_exp_AdamW.log 

source activate $HOME/huckel-jax 
/h/rvargas/.conda/envs/huckel-jax/bin/python main.py --N 10 --lr 2E-2 --l 0 --batch_size 128 --job opt --beta exp -Wdecay h_x h_xy
/h/rvargas/.conda/envs/huckel-jax/bin/python main.py --N 10 --l 0 --job pred --beta exp 



