#! /bin/bash

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF -O --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego -O --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/fox --workspace trial_tensoRF_cp --cp -O --gui
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/nerf_synthetic/lego --workspace trial_tensoRF_lego_cp --cp -O --bound 1.0 --scale 0.8 --dt_gamma 0 --mode blender --gui

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py data/figure --workspace trial_tensoRF_fig -O --cp --gui --scale 0.33 --bound 1.0