# model_type: unet / ffc (需與訓練時一致)

python test.py \
    --data_path datasets/carpet_gray \
    --model_path checkpoints/model_final.pth \
    --output_path outputs \
    --defect_type metal_contamination \
    --img_size 256 \
    --base_channels 64 \
    --model_type ffc \
    --dilate_size 10 \
    --border_margin 10 \
    --vmax 1.0
