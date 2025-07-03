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

## 生產環境問題與解決方案

### 問題描述
在生產環境中發現判別網路對小尺寸或不明顯的異常不敏感，即使重建網路已經正確地將異常部分重建為正常，判別網路卻無法識別出差異。

### 核心原因
判別網路在訓練時學習的是「哪裡有 Perlin 噪聲」而不是「哪裡有差異」。當真實異常的模式（尺寸、形狀、紋理）不符合訓練時的合成異常時，判別網路會忽略它們。

### 解決方案

#### 方案 1：改變訓練策略 - 加入差異圖作為額外監督信號
修改判別網路的輸入，直接將重建差異圖納入考量：

```python
# 在 train_DRAEM.py 的訓練循環中
# 計算重建差異圖
recon_diff = torch.abs(gray_rec - aug_gray_batch)

# 方法 1：直接將差異圖加入判別網路的輸入
joined_in = torch.cat((gray_rec, aug_gray_batch, recon_diff), dim=1)

# 或方法 2：訓練判別網路同時預測差異和異常
# 需要修改判別網路輸出 4 個通道而非 2 個
```

#### 方案 2：多樣化合成異常的生成策略
在 `data_loader.py` 中增加更多異常類型，不只依賴 Perlin 噪聲：

```python
def augment_image(self, image, anomaly_source_path=None):
    # 增加更多異常類型
    anomaly_type = np.random.choice(['perlin', 'scratch', 'spot', 'texture', 'edge'])
    
    if anomaly_type == 'scratch':
        # 生成細線型異常
        mask = self.generate_scratch_mask(image.shape)
    elif anomaly_type == 'spot':
        # 生成小點型異常
        mask = self.generate_spot_mask(image.shape)
    elif anomaly_type == 'edge':
        # 生成邊緣型異常
        mask = self.generate_edge_mask(image.shape)
    else:
        # 原有的 Perlin 噪聲
        mask = self.generate_perlin_mask(image.shape)
```

實作範例：
- **scratch**：使用 cv2.line 生成隨機細線
- **spot**：使用 cv2.circle 生成隨機小圓點
- **edge**：沿著物體邊緣生成異常

#### 方案 3：增強判別網路對細微差異的敏感度
修改損失函數，加入差異感知損失：

```python
class DifferenceAwareLoss(nn.Module):
    """結合 Focal Loss 和差異感知損失"""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        
    def forward(self, pred_mask, target_mask, reconstruction, original):
        # 原有的 Focal Loss
        focal = self.focal_loss(pred_mask, target_mask)
        
        # 計算重建差異
        diff_map = torch.abs(reconstruction - original).mean(dim=1, keepdim=True)
        
        # 差異區域應該被預測為異常
        diff_binary = (diff_map > 0.1).float()  # 閾值可調
        
        # 計算差異感知損失
        diff_loss = F.binary_cross_entropy_with_logits(
            pred_mask[:, 1:2, :, :],  # 異常通道
            diff_binary
        )
        
        # 組合損失
        return focal + 0.5 * diff_loss
```

#### 方案 4：測試時增強策略（無需重新訓練）
在 `test_DRAEM.py` 中加入後處理，結合重建差異增強檢測：

```python
def enhance_detection(anomaly_map, reconstruction, original, sensitivity=1.5):
    """結合重建差異增強異常檢測"""
    
    # 計算重建差異
    diff_map = np.abs(reconstruction - original).mean(axis=2)
    
    # 正規化差異圖
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    
    # 結合原始異常圖和差異圖
    enhanced_map = anomaly_map * 0.7 + diff_map * 0.3
    
    # 增強小異常區域
    enhanced_map = enhanced_map ** (1 / sensitivity)
    
    return enhanced_map
```

#### 方案 5：調整 Perlin 噪聲參數以生成更多樣化的異常
基於 Perlin 噪聲生成的現有參數，調整以產生更符合實際異常的模式：

```python
def augment_image_with_adaptive_perlin(self, image, anomaly_source_path=None):
    aug = self.randAugmenter()
    
    # 動態調整 Perlin 參數以生成不同大小的異常
    anomaly_size_type = np.random.choice(['tiny', 'small', 'medium', 'large', 'mixed'])
    
    if anomaly_size_type == 'tiny':
        # 極小異常：高頻率、高閾值
        perlin_scale_range = (5, 7)  # 2^5 到 2^7，產生更細緻的噪聲
        threshold = 0.7  # 更高閾值，產生更小的異常區域
        beta_range = (0.3, 0.6)  # 較低的混合強度
    elif anomaly_size_type == 'small':
        # 小異常
        perlin_scale_range = (4, 6)
        threshold = 0.65
        beta_range = (0.4, 0.7)
    elif anomaly_size_type == 'medium':
        # 中等異常
        perlin_scale_range = (2, 5)
        threshold = 0.55
        beta_range = (0.5, 0.8)
    elif anomaly_size_type == 'large':
        # 大異常（原始設定）
        perlin_scale_range = (0, 6)
        threshold = 0.5
        beta_range = (0.1, 0.8)
    else:  # mixed
        # 混合多個不同尺度的 Perlin 噪聲
        return self.generate_multi_scale_anomaly(image, anomaly_source_path)
    
    # 生成 Perlin 噪聲
    min_scale, max_scale = perlin_scale_range
    perlin_scalex = 2 ** np.random.randint(min_scale, max_scale)
    perlin_scaley = 2 ** np.random.randint(min_scale, max_scale)
    
    perlin_noise = rand_perlin_2d_np(
        (self.resize_shape[0], self.resize_shape[1]), 
        (perlin_scalex, perlin_scaley)
    )
    
    # 可選：使用不同的 fade 函數改變噪聲特性
    if anomaly_size_type in ['tiny', 'small']:
        # 使用更陡峭的 fade 函數產生更銳利的邊緣
        fade_func = lambda t: t ** 3  # 更簡單的函數，產生更銳利的過渡
        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]), 
            (perlin_scalex, perlin_scaley),
            fade=fade_func
        )
    
    # 應用閾值
    perlin_thr = np.where(perlin_noise > threshold, 
                         np.ones_like(perlin_noise), 
                         np.zeros_like(perlin_noise))
    
    # 對小異常應用形態學操作
    if anomaly_size_type in ['tiny', 'small']:
        from scipy.ndimage import binary_erosion, binary_dilation
        # 隨機選擇形態學操作
        morph_op = np.random.choice(['erosion', 'dilation', 'none'])
        if morph_op == 'erosion':
            kernel = np.ones((3, 3))
            perlin_thr = binary_erosion(perlin_thr, kernel)
        elif morph_op == 'dilation':
            kernel = np.ones((2, 2))
            perlin_thr = binary_dilation(perlin_thr, kernel)
    
    # 調整混合參數
    beta = np.random.uniform(*beta_range)
    
    # ... 後續處理與原始代碼相同 ...

def generate_multi_scale_anomaly(self, image, anomaly_source_path):
    """生成多尺度混合異常"""
    # 生成 2-3 個不同尺度的 Perlin 噪聲並組合
    num_scales = np.random.randint(2, 4)
    combined_mask = np.zeros((self.resize_shape[0], self.resize_shape[1]))
    
    for i in range(num_scales):
        scale = 2 ** np.random.randint(1, 7)
        noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]), 
            (scale, scale)
        )
        threshold = np.random.uniform(0.4, 0.7)
        mask = (noise > threshold).astype(float)
        
        # 隨機權重組合
        weight = np.random.uniform(0.3, 0.7)
        combined_mask = np.maximum(combined_mask, mask * weight)
    
    # 二值化最終遮罩
    final_mask = (combined_mask > 0.5).astype(float)
    
    # ... 後續處理 ...
```

關鍵參數調整說明：
1. **perlin_scale_range**：控制噪聲頻率，更高的值產生更細緻的異常
2. **threshold**：控制異常區域大小，更高的閾值產生更小的異常
3. **beta_range**：控制異常強度，影響異常的可見度
4. **fade 函數**：控制噪聲邊緣的平滑度，可產生更銳利或更平滑的異常邊界
5. **多尺度組合**：混合不同頻率的噪聲，產生更複雜的異常模式

### 實施建議

1. **優先順序**：
   - 立即實施：方案 4（測試時增強）- 無需重新訓練
   - 短期改進：方案 5（調整 Perlin 參數）+ 方案 2（多樣化異常類型）
   - 長期優化：方案 1（改變訓練策略）+ 方案 3（新的損失函數）

2. **組合使用**：
   - 方案 2 + 5 可以一起實施，提供最大的異常多樣性
   - 方案 1 + 3 可以結合，從根本上改善判別網路的學習目標
   - 方案 4 可以作為所有其他方案的補充

3. **參數調優建議**：
   - 根據實際生產環境的異常類型分佈，調整各種異常類型的採樣概率
   - 記錄測試結果，找出最適合特定應用的參數組合
   - 考慮為不同產品類型使用不同的參數設定

### 方案 4 實作說明（已整合至專案）

#### 使用方法

測試時可以使用以下命令啟用差異融合：

```bash
# 使用智能融合方法（區分結構性差異和真實異常）
python test_DRAEM.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" \
    --image_path ./test_image.jpg \
    --fusion_method intelligent \
    --save_debug

# 使用頻率分析融合（區分低頻結構和高頻異常）
python test_DRAEM.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" \
    --test_dir ./RSEM_dataset/test/ \
    --fusion_method frequency

# 使用自適應融合（簡化版但有效）
python test_DRAEM.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" \
    --test_dir ./RSEM_dataset/test/ \
    --fusion_method adaptive

# 不使用融合（預設）
python test_DRAEM.py --model_name "DRAEM_test_0.0001_700_bs8_RSEM_" \
    --test_dir ./RSEM_dataset/test/ \
    --fusion_method none
```

#### 參數說明

- `--fusion_method`: 選擇融合方法
  - `none`: 不使用融合（預設）
  - `intelligent`: 智能融合，使用邊緣檢測和局部統計分析
  - `frequency`: 頻率分析融合，區分低頻和高頻成分
  - `adaptive`: 自適應融合，簡化但有效的方法
- `--save_debug`: 保存除錯圖片（如邊緣遮罩、結構遮罩等）

#### 三種融合方法比較

1. **intelligent（智能融合）**：
   - 優點：最全面，能有效區分結構邊緣和異常
   - 缺點：計算量較大
   - 適用：當異常出現在邊緣附近時

2. **frequency（頻率分析）**：
   - 優點：對紋理異常特別有效
   - 缺點：可能對大型異常不夠敏感
   - 適用：紋理類產品的異常檢測

3. **adaptive（自適應）**：
   - 優點：計算快速，效果穩定
   - 缺點：功能相對簡單
   - 適用：一般用途，快速測試

#### 實作細節

所有融合方法都實作在 `diff_fusion.py` 中，主要功能包括：
- 邊緣抑制：降低結構邊緣的差異權重
- 局部正規化：使異常相對於背景更突出
- 選擇性增強：只在判別器已有響應的區域增強差異

這種設計保持了原有測試流程不變，只是在異常圖生成後加入了可選的增強步驟。