#!/bin/bash

python recon_visualize.py \
    --model_path ./checkpoints/recon_vae_l2+ssim_lr0.0001_ep100_bs8_ch1_256x256_bw64.pth \
    --test_dir ./datasets/testing_ebi_raw_img_256/carpet/test \
    --output_dir ./outputs/recon_vis_vae_bw64/ \
    --gpu_id 0
