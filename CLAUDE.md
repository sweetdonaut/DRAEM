# Claude 助手使用指南

## 語言設定
- 請使用**繁體中文**與使用者對話
- 所有回應都應該使用繁體中文
- 程式碼註解和文件可依需求使用英文或繁體中文

## DRAEM 專案相關資訊

### 專案概述
DRAEM (Discriminatively trained Reconstruction Embedding for surface Anomaly detection) 是一個用於工業表面缺陷檢測的深度學習系統。

### 主要檔案結構
- `train_DRAEM.py`: 訓練腳本
- `test_DRAEM.py`: 測試評估腳本
- `model_unet.py`: 神經網路架構定義
- `data_loader.py`: 資料載入與增強
- `loss.py`: 損失函數實現
- `perlin.py`: Perlin 噪聲生成

### 常用指令
- 訓練模型（使用 DTD）：`python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 700 --data_path ./RSEM_dataset --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/`
- 訓練模型（使用隨機噪聲）：`python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 700 --data_path ./RSEM_dataset --checkpoint_path ./checkpoints/`
- 測試模型：`python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_test_0.0001_700_bs8_RSEM_" --data_path ./RSEM_dataset/`

### 注意事項
- 專案使用 PyTorch 1.8.0 和 CUDA 10.2
- 訓練時只需要正常（無缺陷）的樣本
- 使用 Perlin 噪聲生成合成異常進行訓練
- anomaly_source_path 參數現在是可選的：
  - 提供路徑時：使用外部紋理資料集（如 DTD）
  - 不提供時：使用隨機噪聲生成異常紋理

### 最近的修正
- 修正了 model_unet.py 中的池化層 bug（mp2 = self.mp3 → self.mp2）
- 讓 DTD 資料集變成可選的，適合無法取得外部資料集的生產環境
- 移除了 TensorBoard 依賴，改為直接在終端顯示訓練進度
  - 每 10 個 batch 顯示當前的 loss 值和學習率
  - 每個 epoch 結束時顯示平均 loss 統計
- 將模型儲存格式從 .pckl 改為 .pth
- 在測試時自動儲存視覺化結果，包含：
  - 三張圖片橫向排列：原圖、重建圖、純異常熱力圖（不疊加原圖）
  - 儲存路徑：./outputs/test_results/[obj_name]/[timestamp]/
- 新增 visualize_results.py 用於單獨生成視覺化結果

### 視覺化工具使用方法

#### 使用 visualize.sh 統一視覺化腳本
```bash
# 查看使用說明
./visualize.sh -h

# 列出可用的模型和類別
./visualize.sh -l

# 使用最新模型，處理所有類別
./visualize.sh -a

# 指定模型，處理特定類別
./visualize.sh -m DRAEM_test_0.0001_700_bs8_RSEM_ -c bent -c broken

# 處理單一類別，每類別5張圖
./visualize.sh -c good -n 5

# 完整參數範例
./visualize.sh -m DRAEM_test_0.0001_700_bs8_RSEM_ -c bent -c glue -n 20 -g 1
```

參數說明：
- `-m, --model`: 指定模型名稱（不含 .pth）
- `-c, --category`: 指定類別（可多次使用）
- `-a, --all`: 處理所有類別
- `-n, --num`: 每個類別最多處理幾張圖片（預設：10）
- `-g, --gpu`: 指定 GPU ID（預設：0）
- `-l, --list`: 列出可用的模型和類別
- `-h, --help`: 顯示說明

#### 使用 Python 腳本（進階用法）
- 視覺化單張圖片：`python visualize_results.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" --image_path ./test_image.jpg`
- 視覺化測試資料夾：`python visualize_results.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" --test_dir ./RSEM_dataset/test/`