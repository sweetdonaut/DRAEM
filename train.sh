#!/bin/bash
# 簡單的 DRAEM 訓練腳本

# 預設值
CHANNELS=1
IMG_HEIGHT=256
IMG_WIDTH=256

# 解析參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --img_size)
            IMG_HEIGHT="$2"
            IMG_WIDTH="$3"
            shift 3
            ;;
        *)
            echo "未知參數：$1"
            echo "使用方法：./train.sh [--channels 1|3] [--img_size HEIGHT WIDTH]"
            echo "範例：./train.sh --channels 1 --img_size 976 176"
            exit 1
            ;;
    esac
done

# 訓練模型
echo "開始訓練..."
echo "通道數：$CHANNELS"
echo "圖片尺寸：${IMG_HEIGHT}x${IMG_WIDTH}"
python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 2 --data_path ./RSEM_dataset --checkpoint_path ./checkpoints/ --channels $CHANNELS --img_size $IMG_HEIGHT $IMG_WIDTH

# 檢查訓練是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "訓練完成！"
else
    echo "訓練失敗！"
    exit 1
fi