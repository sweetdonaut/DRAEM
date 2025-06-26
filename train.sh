#!/bin/bash
# 簡單的 DRAEM 訓練腳本

# 預設值
CHANNELS=3

# 解析參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        *)
            echo "未知參數：$1"
            echo "使用方法：./train.sh [--channels 1|3]"
            exit 1
            ;;
    esac
done

# 訓練模型
echo "開始訓練..."
echo "通道數：$CHANNELS"
python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 700 --data_path ./RSEM_dataset --checkpoint_path ./checkpoints/ --channels $CHANNELS

# 檢查訓練是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "訓練完成！開始測試 bent 類別..."
    echo ""
    
    # 自動執行測試，選擇 bent 類別
    ./test.sh -c bent -n 10
else
    echo "訓練失敗！"
    exit 1
fi