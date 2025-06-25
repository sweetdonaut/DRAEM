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
- 訓練模型（使用 DTD）：`python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 700 --data_path ./RSEM_dataset --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/ --visualize`
- 訓練模型（使用隨機噪聲）：`python train_DRAEM.py --gpu_id 0 --lr 0.0001 --bs 8 --epochs 700 --data_path ./RSEM_dataset --checkpoint_path ./checkpoints/ --log_path ./logs/ --visualize`
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