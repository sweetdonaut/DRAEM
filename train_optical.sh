#!/bin/bash
# OpticalDatasetSlide 訓練腳本 (128x128 patches)

echo "=========================================="
echo "訓練 OpticalDatasetSlide (128x128 patches)"
echo "使用專用腳本 train_DRAEM_optical.py"
echo "=========================================="
echo ""

# 預設值
CHANNELS=1
IMG_HEIGHT=128
IMG_WIDTH=128
BATCH_SIZE=8
EPOCHS=1
LR=0.0001
GPU_ID=0
NUM_WORKERS=4

# 顯示參數
echo "訓練參數:"
echo "- 通道數: $CHANNELS"
echo "- 圖片尺寸: ${IMG_HEIGHT}x${IMG_WIDTH}"
echo "- Batch size: $BATCH_SIZE"
echo "- Epochs: $EPOCHS"
echo "- Learning rate: $LR"
echo "- GPU ID: $GPU_ID"
echo "- Workers: $NUM_WORKERS"
echo ""

# 訓練模型
echo "開始訓練..."
python train_DRAEM_optical.py \
    --gpu_id $GPU_ID \
    --lr $LR \
    --bs $BATCH_SIZE \
    --epochs $EPOCHS \
    --data_path ./OpticalDatasetSlide \
    --checkpoint_path ./checkpoints/ \
    --channels $CHANNELS \
    --img_size $IMG_HEIGHT $IMG_WIDTH \
    --num_workers $NUM_WORKERS

# 檢查訓練是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 訓練完成！"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 訓練失敗！"
    echo "=========================================="
    exit 1
fi