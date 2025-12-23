# model_type: unet / ffc
# loss_type: l2 / ssim / both

python train.py \
    --data_path datasets/carpet_gray \
    --save_path checkpoints \
    --img_size 256 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --base_channels 64 \
    --model_type ffc \
    --loss_type ssim \
    --dilate_size 10 \
    --border_margin 10 \
    --num_workers 4
