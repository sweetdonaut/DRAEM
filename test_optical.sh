#!/bin/bash

# DRAEM OpticalDataset 滑動視窗測試腳本
# 支援選擇模型、選擇測試類別、批次處理、融合方法選擇

# 預設值
DATA_PATH="./OpticalDatasetSlide/test"
CHECKPOINT_PATH="./checkpoints"
GPU_ID=0
MAX_IMAGES=10
CUSTOM_DATA_PATH=""
MERGE_METHOD="both"  # max, average, or both

# 顯示使用說明
show_help() {
    echo "DRAEM OpticalDataset 滑動視窗測試工具"
    echo ""
    echo "使用方法："
    echo "  ./test_optical.sh [選項]"
    echo ""
    echo "選項："
    echo "  -m, --model MODEL_NAME     指定模型名稱（不含 .pth）"
    echo "  -c, --category CATEGORY    指定類別（可多次使用），例如：-c bent -c broken"
    echo "  -a, --all                  處理所有類別"
    echo "  -n, --num NUM              每個類別最多處理幾張圖片（預設：10）"
    echo "  -g, --gpu GPU_ID           指定 GPU ID（預設：0）"
    echo "  -d, --data DATA_PATH       指定資料集路徑（預設：./OpticalDatasetSlide/test）"
    echo "  -f, --fusion METHOD        融合方法：max, average, both（預設：both）"
    echo "  -l, --list                 列出可用的模型和類別"
    echo "  -h, --help                 顯示此說明"
    echo ""
    echo "範例："
    echo "  # 列出可用的模型和類別"
    echo "  ./test_optical.sh -l"
    echo ""
    echo "  # 使用最新模型，處理所有類別"
    echo "  ./test_optical.sh -a"
    echo ""
    echo "  # 指定模型，處理特定類別"
    echo "  ./test_optical.sh -m DRAEM_optical_0.0001_10_bs8_ch1_128x128 -c bent -c broken"
    echo ""
    echo "  # 處理單一類別，每類別5張圖，只用最大值融合"
    echo "  ./test_optical.sh -c good -n 5 -f max"
    echo ""
    echo "  # 處理所有類別，使用兩種融合方法"
    echo "  ./test_optical.sh -a -f both"
}

# 列出可用資源
list_resources() {
    echo "========== 可用的 OpticalDataset 模型 =========="
    if [ -d "$CHECKPOINT_PATH" ]; then
        models=$(ls -1 "$CHECKPOINT_PATH"/DRAEM_optical_*.pth 2>/dev/null | grep -v "_seg" | sed 's/.*\///g' | sed 's/\.pth$//')
        if [ -z "$models" ]; then
            echo "（無）"
        else
            echo "$models" | nl -w2 -s'. '
            # 找出最新的模型
            latest=$(ls -t "$CHECKPOINT_PATH"/DRAEM_optical_*.pth 2>/dev/null | grep -v "_seg" | head -1 | sed 's/.*\///g' | sed 's/\.pth$//')
            echo ""
            echo "最新模型：$latest"
        fi
    else
        echo "找不到 checkpoint 目錄"
    fi
    
    echo ""
    echo "========== 可用的測試類別 =========="
    if [ -d "$DATA_PATH" ]; then
        categories=$(ls -1 "$DATA_PATH" 2>/dev/null)
        if [ -z "$categories" ]; then
            echo "（無）"
        else
            echo "$categories" | nl -w2 -s'. '
        fi
    else
        echo "找不到測試資料目錄"
    fi
}

# 解析命令列參數
MODEL_NAME=""
CATEGORIES=()
PROCESS_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORIES+=("$2")
            shift 2
            ;;
        -a|--all)
            PROCESS_ALL=true
            shift
            ;;
        -n|--num)
            MAX_IMAGES="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -d|--data)
            CUSTOM_DATA_PATH="$2"
            shift 2
            ;;
        -f|--fusion)
            MERGE_METHOD="$2"
            shift 2
            ;;
        -l|--list)
            list_resources
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知選項：$1"
            echo "使用 -h 或 --help 查看說明"
            exit 1
            ;;
    esac
done

# 如果有自訂資料路徑，使用它
if [ -n "$CUSTOM_DATA_PATH" ]; then
    DATA_PATH="$CUSTOM_DATA_PATH"
fi

# 如果沒有指定模型，使用最新的 optical 模型
if [ -z "$MODEL_NAME" ]; then
    LATEST_MODEL=$(ls -t "$CHECKPOINT_PATH"/DRAEM_optical_*.pth 2>/dev/null | grep -v "_seg" | head -1)
    if [ -z "$LATEST_MODEL" ]; then
        echo "錯誤：找不到任何 OpticalDataset 模型檔案"
        echo "請先使用 train_optical.sh 訓練模型或使用 -m 指定模型名稱"
        exit 1
    fi
    MODEL_NAME=$(basename "$LATEST_MODEL" .pth)
    echo "使用最新模型：$MODEL_NAME"
fi

# 檢查模型是否存在
if [ ! -f "$CHECKPOINT_PATH/${MODEL_NAME}.pth" ]; then
    echo "錯誤：找不到模型檔案 $CHECKPOINT_PATH/${MODEL_NAME}.pth"
    echo "使用 -l 查看可用的模型"
    exit 1
fi

# 決定要處理的類別
if [ "$PROCESS_ALL" = true ]; then
    # 處理所有類別
    CATEGORIES=($(ls -1 "$DATA_PATH" 2>/dev/null))
elif [ ${#CATEGORIES[@]} -eq 0 ]; then
    # 沒有指定類別，顯示說明
    echo "請指定要處理的類別（使用 -c）或使用 -a 處理所有類別"
    echo "使用 -l 查看可用的類別"
    exit 1
fi

# 驗證融合方法
if [[ "$MERGE_METHOD" != "max" && "$MERGE_METHOD" != "average" && "$MERGE_METHOD" != "both" ]]; then
    echo "錯誤：無效的融合方法 '$MERGE_METHOD'"
    echo "請使用 max, average 或 both"
    exit 1
fi

# 開始處理
echo ""
echo "=========================================="
echo "DRAEM OpticalDataset 滑動視窗測試"
echo "模型：$MODEL_NAME"
echo "類別：${CATEGORIES[*]}"
echo "每類別最多：$MAX_IMAGES 張圖片"
echo "GPU ID：$GPU_ID"
echo "融合方法：$MERGE_METHOD"
echo "=========================================="
echo ""

# 創建輸出目錄
timestamp=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/visualizations/${MODEL_NAME}_${timestamp}"

# 處理每個類別
for category in "${CATEGORIES[@]}"; do
    category_path="$DATA_PATH/$category"
    
    if [ ! -d "$category_path" ]; then
        echo "警告：找不到類別目錄 $category_path，跳過..."
        continue
    fi
    
    echo "處理類別：$category"
    
    # 取得該類別的前 N 張圖片（支援 .tiff）
    images=($(ls "$category_path"/*.tiff 2>/dev/null | head -$MAX_IMAGES))
    
    if [ ${#images[@]} -eq 0 ]; then
        echo "  沒有找到圖片，跳過..."
        continue
    fi
    
    echo "  找到 ${#images[@]} 張圖片"
    
    # 創建類別輸出目錄
    category_output="$OUTPUT_DIR/$category"
    mkdir -p "$category_output"
    
    # 處理每張圖片
    for img_path in "${images[@]}"; do
        img_name=$(basename "$img_path")
        echo "  處理 $img_name ..."
        
        # 根據融合方法設定參數
        if [[ "$MERGE_METHOD" == "both" ]]; then
            merge_arg=""
        else
            merge_arg="--merge_method $MERGE_METHOD"
        fi
        
        # 執行滑動視窗測試
        python test_DRAEM_optical_slide.py \
            --model_name "$MODEL_NAME" \
            --image_path "$img_path" \
            --output_dir "$category_output" \
            --gpu_id $GPU_ID \
            --test_sliding_window \
            $merge_arg \
            2>&1 | while IFS= read -r line; do
                # 過濾並格式化輸出
                if [[ "$line" == *"✓ 滑動視窗推論測試成功"* ]]; then
                    echo "    ✓ 推論成功"
                elif [[ "$line" == *"提取了"*"個 patches"* ]]; then
                    echo "    $line" | sed 's/^[ ]*/    /'
                elif [[ "$line" == *"異常分數範圍"* ]]; then
                    echo "    $line" | sed 's/^[ ]*/    /'
                elif [[ "$line" == *"平均異常分數"* ]]; then
                    echo "    $line" | sed 's/^[ ]*/    /'
                elif [[ "$line" == *"結果儲存至"* ]]; then
                    # 根據融合方法決定是否顯示
                    if [[ "$MERGE_METHOD" == "both" ]]; then
                        echo "    $line" | sed 's/^[ ]*/    /'
                    elif [[ "$MERGE_METHOD" == "max" && "$line" == *"_max.png"* ]]; then
                        echo "    $line" | sed 's/^[ ]*/    /'
                    elif [[ "$MERGE_METHOD" == "average" && "$line" == *"_average.png"* ]]; then
                        echo "    $line" | sed 's/^[ ]*/    /'
                    fi
                elif [[ "$line" == *"✗ 滑動視窗推論測試失敗"* ]]; then
                    echo "    ✗ 推論失敗"
                    echo "    $line" | sed 's/^[ ]*/    /'
                fi
            done
        
        echo ""
    done
    
    echo "  結果儲存在：$category_output"
    echo ""
done

echo "=========================================="
echo "視覺化完成！"
echo "所有結果儲存在：$OUTPUT_DIR"
echo "=========================================="