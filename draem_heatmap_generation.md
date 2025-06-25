# Draem 模型 Heatmap 生成流程詳解

## 目錄
1. [概述](#概述)
2. [技術架構](#技術架構)
3. [Anomaly Map 生成流程](#anomaly-map-生成流程)
4. [Pred Mask 生成流程](#pred-mask-生成流程)
5. [後處理機制](#後處理機制)
6. [實作範例](#實作範例)
7. [視覺化流程圖](#視覺化流程圖)

## 概述

Draem (DRÆM - Discriminatively trained Reconstruction Anomaly Embedding Model) 是一種基於重建的異常偵測模型。本文檔詳細說明該模型如何生成兩個關鍵輸出：

- **Anomaly Map（異常熱力圖）**：顯示每個像素為異常的機率，數值範圍 0-1
- **Pred Mask（預測遮罩）**：二值化的異常區域標記，只有 0（正常）或 1（異常）

## 技術架構

### Draem 模型詳細架構

DRAEM (Discriminatively trained Reconstruction Autoencoder for surface anomaly detection) 採用創新的雙分支架構，包含重建子網路和判別子網路。

#### 1. 重建子網路（Reconstructive Subnetwork）

**編碼器結構**：
```python
# EncoderReconstructive
- 輸入: [B, 3, H, W] (RGB 圖像)
- Block 1: Conv(3→128) → Conv(128→128) → MaxPool
- Block 2: Conv(128→256) → Conv(256→256) → MaxPool
- Block 3: Conv(256→512) → Conv(512→512) → MaxPool
- Block 4: Conv(512→512) → Conv(512→512) → MaxPool
- Block 5: Conv(512→1024) → Conv(1024→1024) + 可選 SSPCAB
- 輸出: [B, 1024, H/32, W/32]
```

**解碼器結構**：
```python
# DecoderReconstructive
- Block 4: Upsample → Conv(1024→512) → Conv(512→512)
- Block 3: Upsample → Conv(512→256) → Conv(256→256)
- Block 2: Upsample → Conv(256→128) → Conv(128→128)
- Block 1: Upsample → Conv(128→128) → Conv(128→64)
- Output: Conv(64→3) → Tanh
- 輸出: [B, 3, H, W] (重建圖像)
```

#### 2. 判別子網路（Discriminative Subnetwork）

**編碼器結構（含跳躍連接）**：
```python
# EncoderDiscriminative
- 輸入: [B, 6, H, W] (原圖 + 重建圖拼接)
- 返回所有 6 個塊的激活值用於跳躍連接
- 每個塊的通道數逐漸增加: 64→128→256→512→512→512
```

**解碼器結構（使用跳躍連接）**：
```python
# DecoderDiscriminative
- 使用編碼器的跳躍連接進行特徵融合
- 最終輸出: Conv(→2) 生成二分類結果
- 輸出: [B, 2, H, W] (正常/異常的 logit 值)
```

#### 3. SSPCAB 模組（可選）

Self-Supervised Predictive Convolutional Attention Block 提供額外的自監督信號：

```python
class SSPCAB:
    def __init__(self):
        # 4 個方向的遮罩卷積 (top, left, bottom, right)
        self.masked_conv = MaskedConv2d(directions=4)
        # 通道注意力機制
        self.channel_attention = ChannelAttention()
    
    def forward(self, x):
        # 遮罩卷積預測
        predictions = self.masked_conv(x)
        # 通道注意力加權
        weighted = self.channel_attention(x) * x
        return weighted, predictions
```

### Draem 模型輸出結構

經過雙分支處理後，最終輸出為 2 通道的特徵圖：

```python
# 訓練時返回兩個輸出
reconstruction, prediction = model(input_image)
# reconstruction: [B, 3, H, W] - 重建圖像
# prediction: [B, 2, H, W] - 異常分類 logit

# 推論時直接處理為異常圖
anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
```

## 訓練過程詳解

### 合成異常生成（Synthetic Anomaly Generation）

DRAEM 的核心創新之一是使用合成異常進行訓練。這透過 **PerlinAnomalyGenerator** 實現：

#### 1. Perlin 噪聲生成

```python
def generate_perlin_noise(shape, scale=6):
    # 生成梯度向量
    gradients = generate_gradient_vectors()
    
    # 對每個像素計算 Perlin 噪聲值
    for y in range(height):
        for x in range(width):
            # 雙線性插值
            noise_value = bilinear_interpolation(x, y, gradients)
            # 使用 fade 函數平滑過渡
            noise_value = fade(noise_value)
    
    return noise_map
```

#### 2. 異常遮罩生成

```python
# 生成二值化遮罩
perlin_noise = generate_perlin_noise(image.shape)
mask = (perlin_noise > 0.5).astype(float)

# 應用形態學操作使遮罩更自然
mask = morphological_operations(mask)
```

#### 3. 異常混合策略

```python
def augment_image(image, anomaly_source):
    # 數據增強
    if random.random() > 0.5:
        anomaly_source = color_jitter(anomaly_source)
    if random.random() > 0.5:
        anomaly_source = sharpen(anomaly_source)
    
    # 混合因子
    beta = random.uniform(0.1, 1.0)
    
    # 生成合成異常圖像
    augmented = image * (1 - mask) + beta * anomaly_source + (1 - beta) * image * mask
    
    return augmented, mask
```

### 損失函數設計

DRAEM 使用多個損失函數的組合來訓練模型：

#### 1. 重建損失（L2 Loss）

```python
# 確保重建圖像接近原始圖像
l2_loss = F.mse_loss(reconstruction, original_image)
```

#### 2. 結構相似性損失（SSIM Loss）

```python
# 保持結構信息
ssim_loss = 1 - ssim(reconstruction, original_image, window_size=11)
ssim_loss_weighted = 2 * ssim_loss  # 權重為 2
```

#### 3. 焦點損失（Focal Loss）

```python
# 處理類別不平衡問題
focal_loss = FocalLoss(alpha=1, reduction='mean')
focal_loss_value = focal_loss(prediction, target_mask)
```

#### 4. SSPCAB 損失（可選）

```python
if enable_sspcab:
    # 自監督預測損失
    sspcab_loss = F.mse_loss(predicted_features, target_features)
    total_loss += sspcab_lambda * sspcab_loss
```

#### 5. 總損失

```python
total_loss = l2_loss + 2 * ssim_loss + focal_loss + sspcab_lambda * sspcab_loss
```

### 訓練策略

1. **優化器配置**：
```python
optimizer = Adam(model.parameters(), lr=0.0001)
scheduler = MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)
```

2. **訓練循環**：
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 生成合成異常
        augmented, masks = anomaly_generator(batch)
        
        # 前向傳播
        reconstruction, prediction = model(augmented)
        
        # 計算損失
        loss = compute_loss(reconstruction, augmented, prediction, masks)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Anomaly Map 生成流程

### 步驟 1：Softmax 轉換

將原始的 logit 輸出轉換為機率值：

```python
# 在 anomalib/models/image/draem/torch_model.py 第 71 行
anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
```

**數學公式**：
```
對於每個像素位置 (i,j)：
P(異常) = exp(logit_異常) / (exp(logit_正常) + exp(logit_異常))
```

### 步驟 2：提取異常通道

只保留第 1 通道（異常類別）的機率值作為 anomaly map：

```python
# anomaly_map 形狀：[batch_size, height, width]
# 數值範圍：[0, 1]
```

### 步驟 3：計算圖片級別異常分數

```python
# 取 anomaly map 中的最大值作為整張圖片的異常分數
pred_score = torch.amax(anomaly_map, dim=(-2, -1))
```

### 完整的推論流程

```python
def forward(self, batch: Tensor) -> InferenceBatch | tuple[Tensor, Tensor]:
    """DRAEM 前向傳播過程"""
    # 1. 重建子網路：生成重建圖像
    reconstruction = self.reconstructive_subnetwork(batch)
    
    # 2. 拼接原圖和重建圖
    concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
    
    # 3. 判別子網路：預測異常區域
    prediction = self.discriminative_subnetwork(concatenated_inputs)
    
    if self.training:
        # 訓練模式：返回原始輸出供損失計算
        return reconstruction, prediction
    else:
        # 推論模式：生成異常圖和分數
        # 使用 softmax 轉換為機率
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
        
        # 計算圖像級別異常分數（最大值策略）
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        
        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map,
        )
```

## Pred Mask 生成流程

### 步驟 1：正規化（可選）

PostProcessor 可以對 anomaly map 進行 min-max 正規化：

```python
# 正規化到 [0, 1] 範圍
if normalize:
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
```

### 步驟 2：閾值計算

有兩種主要的閾值計算方式：

#### 方式 A：自適應閾值（F1AdaptiveThreshold）

```python
# 在驗證集上計算最佳閾值
def compute_adaptive_threshold(anomaly_maps, ground_truths):
    best_threshold = 0
    best_f1 = 0
    
    for threshold in np.linspace(0, 1, 100):
        pred_masks = anomaly_maps > threshold
        precision = calculate_precision(pred_masks, ground_truths)
        recall = calculate_recall(pred_masks, ground_truths)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

#### 方式 B：敏感度閾值

```python
# 使用固定的敏感度參數
sensitivity = 0.5  # 範圍：[0, 1]
threshold = 1.0 - sensitivity
```

### 步驟 3：二值化

應用閾值生成二值化遮罩：

```python
pred_mask = (anomaly_map > threshold).float()
# pred_mask 只包含 0 或 1
```

## 後處理機制

### 形態學操作

為了去除雜訊和平滑結果，可以應用形態學操作：

```python
from scipy.ndimage import binary_opening

# 開運算：先腐蝕後膨脹，去除小的雜訊點
kernel_size = 3
pred_mask = binary_opening(pred_mask, structure=np.ones((kernel_size, kernel_size)))
```

### PostProcessor 類別整合

在 anomalib 中，PostProcessor 統一處理這些步驟：

```python
class PostProcessor:
    def forward(self, predictions):
        # 1. 提取 anomaly map
        anomaly_map = predictions["anomaly_map"]
        
        # 2. 正規化（如果啟用）
        if self.normalize:
            anomaly_map = self._normalize(anomaly_map)
        
        # 3. 應用閾值
        pred_mask = anomaly_map > self.threshold
        
        # 4. 計算圖片級別標籤
        pred_label = torch.any(pred_mask.view(batch_size, -1), dim=1)
        
        return {
            "anomaly_map": anomaly_map,
            "pred_mask": pred_mask,
            "pred_label": pred_label,
            "pred_score": pred_score
        }
```

## 實作範例

### 存取 Anomaly Map 和 Pred Mask

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Draem

# 初始化模型和資料
model = Draem()
datamodule = MVTecAD(root="./datasets/MVTecAD", category="grid")
engine = Engine()

# 執行推論
predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path="path/to/checkpoint.ckpt")

# 存取結果
for batch in predictions:
    batch_size = batch.pred_label.shape[0]
    for i in range(batch_size):
        # 取得 anomaly map（連續值）
        anomaly_map = batch.anomaly_map[i].cpu().numpy()  # shape: [H, W]
        
        # 取得 pred mask（二值化）
        pred_mask = batch.pred_mask[i].cpu().numpy()  # shape: [H, W]
        
        # 取得異常分數
        anomaly_score = batch.pred_score[i].item()  # 單一數值
        
        print(f"Anomaly map 範圍: [{anomaly_map.min():.3f}, {anomaly_map.max():.3f}]")
        print(f"Pred mask 唯一值: {np.unique(pred_mask)}")
        print(f"異常分數: {anomaly_score:.3f}")
```

### 自訂視覺化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(image, anomaly_map, pred_mask, threshold=0.5):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    # 原始圖片
    axes[0].imshow(image)
    axes[0].set_title("原始圖片")
    axes[0].axis('off')
    
    # Anomaly Map（熱力圖）
    im = axes[1].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Anomaly Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Pred Mask（二值化）
    axes[2].imshow(pred_mask, cmap='binary', vmin=0, vmax=1)
    axes[2].set_title(f"Pred Mask (閾值={threshold})")
    axes[2].axis('off')
    
    # 疊加顯示
    overlay = image.copy()
    mask_indices = pred_mask > 0
    overlay[mask_indices] = [255, 0, 0]  # 紅色標記異常區域
    axes[3].imshow(overlay)
    axes[3].set_title("異常區域標記")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 視覺化流程圖

### DRAEM 雙分支架構流程

```
訓練階段：
┌─────────────────┐
│  正常圖片輸入   │
│  [B, 3, H, W]   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Perlin 異常生成 │
│ + 數據增強      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        合成異常圖片 + 遮罩          │
│  augmented_img    ground_truth_mask │
└────────┬───────────────────┬────────┘
         │                   │
         ▼                   │
┌─────────────────┐          │
│ 重建子網路      │          │
│ Encoder (5層)   │          │
│ + SSPCAB (可選) │          │
│ Decoder (4層)   │          │
└────────┬────────┘          │
         │                   │
         ▼                   │
┌─────────────────┐          │
│   重建圖像      │          │
│  [B, 3, H, W]   │          │
└────────┬────────┘          │
         │                   │
         ▼                   │
┌─────────────────┐          │
│ 拼接原圖+重建圖 │          │
│  [B, 6, H, W]   │          │
└────────┬────────┘          │
         │                   │
         ▼                   │
┌─────────────────┐          │
│ 判別子網路      │          │
│ Encoder (6層)   │          │
│ + 跳躍連接      │          │
│ Decoder (5層)   │          │
└────────┬────────┘          │
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌──────────────┐
│ 預測結果        │  │ 損失計算     │
│ [B, 2, H, W]    │  │ L2 + SSIM    │
│                 │  │ + Focal Loss │
└─────────────────┘  └──────────────┘

推論階段：
┌─────────────────┐
│  測試圖片輸入   │
│  [B, 3, H, W]   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 重建子網路      │
│ (生成重建圖)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 拼接 [原圖|重建]│
│  [B, 6, H, W]   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 判別子網路      │
│ (預測異常區域)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   2通道輸出     │
│ [B, 2, H, W]    │
│ Ch0: 正常 logit │
│ Ch1: 異常 logit │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Softmax 轉換   │
│ P = exp(x)/Σexp │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Anomaly Map    │
│  [B, H, W]      │
│  範圍: [0, 1]   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌────────────┐
│ 閾值 │  │  Max Pool  │
│處理  │  │            │
└──┬───┘  └─────┬──────┘
   │            │
   ▼            ▼
┌──────────┐  ┌────────────┐
│Pred Mask │  │ Pred Score │
│ {0, 1}   │  │  [0, 1]    │
└──────────┘  └────────────┘
```

## 重要參數說明

### PostProcessor 參數
| 參數 | 說明 | 預設值 | 範圍 |
|------|------|--------|------|
| `normalize` | 是否正規化 anomaly map | True | True/False |
| `threshold` | 二值化閾值 | 自適應 | [0, 1] |
| `sensitivity` | 敏感度（用於計算閾值） | 0.5 | [0, 1] |
| `kernel_size` | 形態學操作的核大小 | 3 | 奇數 |

### DRAEM 模型特有參數
| 參數 | 說明 | 預設值 | 範圍 |
|------|------|--------|------|
| `anomaly_source_path` | 異常源圖片路徑 | None | 路徑字串 |
| `enable_sspcab` | 是否啟用 SSPCAB 模組 | False | True/False |
| `sspcab_lambda` | SSPCAB 損失權重 | 0.1 | [0, 1] |
| `beta` | 異常混合強度範圍 | (0.1, 1.0) | [0, 1] |

### 訓練參數
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `learning_rate` | 學習率 | 0.0001 |
| `batch_size` | 批次大小 | 8 |
| `num_epochs` | 訓練輪數 | 700 |
| `milestones` | 學習率衰減點 | [400, 600] |
| `gamma` | 學習率衰減因子 | 0.1 |

## 結論

### DRAEM 模型的核心流程總結

DRAEM 模型通過以下完整流程生成異常偵測結果：

#### 訓練階段
1. **合成異常生成**：使用 Perlin 噪聲生成自然的異常模式
2. **雙分支學習**：
   - 重建子網路學習正常圖像的表示
   - 判別子網路學習區分正常和異常區域
3. **多損失優化**：結合 L2、SSIM、Focal Loss 進行端到端訓練

#### 推論階段
1. **重建生成**：重建子網路生成輸入圖像的重建版本
2. **異常判別**：判別子網路比較原圖和重建圖，輸出 2 通道 logit
3. **Softmax 轉換**：將 logit 轉為機率，生成 anomaly map
4. **閾值處理**：應用自適應或固定閾值生成 pred mask
5. **後處理優化**：形態學操作和正規化提升結果品質

### DRAEM 的關鍵創新

1. **無需真實異常樣本**：透過合成異常進行訓練，解決異常樣本稀缺問題
2. **端到端可訓練**：整合重建和判別於單一架構，簡化訓練流程
3. **精確的像素級定位**：判別網路的跳躍連接保留細節信息
4. **自適應閾值機制**：根據驗證集自動確定最佳閾值
5. **可擴展性**：SSPCAB 等模組可選擇性加入以提升性能

### 實際應用建議

1. **調整敏感度**：
   - 高敏感度（低閾值）：減少漏檢，但可能增加誤檢
   - 低敏感度（高閾值）：減少誤檢，但可能增加漏檢

2. **優化訓練**：
   - 調整 `beta` 範圍控制合成異常的強度
   - 使用真實異常圖片作為 anomaly source 可能提升效果
   - 啟用 SSPCAB 可能提升對複雜紋理的檢測能力

3. **後處理改進**：
   - 根據應用場景調整形態學操作的 kernel size
   - 考慮使用連通區域分析過濾小的誤檢區域
   - 可以結合多個模型的結果進行集成

理解 DRAEM 的完整流程有助於：
- 根據具體需求調整模型配置
- 開發針對特定場景的改進方案
- 整合到實際的品質檢測系統中
- 為新的異常檢測方法提供啟發