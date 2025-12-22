#!/bin/bash

# --model_type: original, skip, vae
# --skip_layers: 2, 3, 4, or combinations like: 2 3, 2 4, 3 4, 2 3 4
# --loss_type: l2, ssim, l2+ssim
# --base_width: 64, 128 (default), 256
# --dtd_path ./datasets/dtd/images/

python recon_train.py \
    --model_type vae \
    --base_width 64 \
    --loss_type l2+ssim \
    --gpu_id 0 \
    --lr 0.0001 \
    --bs 8 \
    --epochs 100 \
    --data_path ./datasets/testing_ebi_raw_img_256/carpet \
    --checkpoint_path ./checkpoints/ \
    --channels 1 \
    --img_size 256 256
