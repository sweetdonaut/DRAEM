"""
差異融合方法集合
用於增強 DRAEM 判別器對小異常的敏感度
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters


def intelligent_diff_fusion(anomaly_map, reconstruction, original, 
                          edge_threshold=0.1, 
                          structure_kernel_size=15,
                          anomaly_kernel_size=5):
    """
    智能融合差異資訊，區分結構性差異和真實異常
    
    Args:
        anomaly_map: 原始異常圖 (H, W)
        reconstruction: 重建圖像 (H, W, C) 或 (H, W)
        original: 原始圖像 (H, W, C) 或 (H, W)
        edge_threshold: 邊緣檢測閾值
        structure_kernel_size: 結構邊緣擴張核大小
        anomaly_kernel_size: 局部異常統計核大小
    
    Returns:
        enhanced_map: 增強後的異常圖
        debug_info: 包含中間結果的字典（用於除錯）
    """
    
    # 確保輸入格式一致
    if len(reconstruction.shape) == 2:
        reconstruction = np.expand_dims(reconstruction, axis=2)
    if len(original.shape) == 2:
        original = np.expand_dims(original, axis=2)
    
    # 1. 計算原始差異
    diff = np.abs(reconstruction - original)
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)
    
    # 2. 識別結構性邊緣（這些是正常的）
    # 轉換為 uint8 進行邊緣檢測
    if original.max() <= 1:
        original_uint8 = (original * 255).astype(np.uint8)
        reconstruction_uint8 = (reconstruction * 255).astype(np.uint8)
    else:
        original_uint8 = original.astype(np.uint8)
        reconstruction_uint8 = reconstruction.astype(np.uint8)
    
    if len(original_uint8.shape) == 3:
        edges_original = cv2.Canny(cv2.cvtColor(original_uint8, cv2.COLOR_RGB2GRAY), 50, 150) / 255.0
        edges_reconstruction = cv2.Canny(cv2.cvtColor(reconstruction_uint8, cv2.COLOR_RGB2GRAY), 50, 150) / 255.0
    else:
        edges_original = cv2.Canny(original_uint8[:,:,0], 50, 150) / 255.0
        edges_reconstruction = cv2.Canny(reconstruction_uint8[:,:,0], 50, 150) / 255.0
    
    # 結構性邊緣 = 兩張圖都有的邊緣
    structural_edges = np.minimum(edges_original, edges_reconstruction)
    
    # 3. 擴張結構邊緣區域（因為重建的邊緣可能有位移）
    struct_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (structure_kernel_size, structure_kernel_size))
    structural_mask = cv2.dilate(structural_edges, struct_kernel, iterations=1)
    
    # 4. 計算非結構性差異
    # 在結構邊緣區域降低差異權重
    non_structural_diff = diff * (1 - structural_mask * 0.8)
    
    # 5. 局部異常檢測
    # 計算局部統計量
    local_mean = ndimage.uniform_filter(non_structural_diff, size=anomaly_kernel_size)
    local_std = ndimage.generic_filter(non_structural_diff, np.std, size=anomaly_kernel_size)
    
    # 異常分數 = 局部標準化後的差異
    # 避免除以零
    anomaly_score = np.where(local_std > 0.01,
                           (non_structural_diff - local_mean) / (local_std + 1e-8),
                           0)
    
    # 6. 自適應閾值處理
    # 只保留顯著高於局部平均的差異
    significant_diff = np.where(anomaly_score > 2.0,  # 2個標準差以上
                               non_structural_diff,
                               non_structural_diff * 0.1)  # 抑制不顯著的差異
    
    # 7. 結合原始 anomaly map
    # 使用動態權重，根據差異的顯著性調整
    diff_confidence = np.clip(anomaly_score / 3.0, 0, 1)  # 轉換為 0-1 信心度
    
    # 智能融合：當差異顯著時，增加其權重
    enhanced_map = (anomaly_map * (1 - diff_confidence * 0.5) + 
                   significant_diff * diff_confidence * 0.5)
    
    # 8. 後處理：增強小異常
    # 使用形態學 top-hat 變換來突出小的亮點
    small_anomaly_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    top_hat = cv2.morphologyEx(enhanced_map, cv2.MORPH_TOPHAT, small_anomaly_kernel)
    enhanced_map = enhanced_map + top_hat * 0.5
    
    # 9. 最終正規化
    enhanced_map = np.clip(enhanced_map, 0, 1)
    
    debug_info = {
        'structural_mask': structural_mask,
        'anomaly_score': anomaly_score,
        'significant_diff': significant_diff,
        'diff_confidence': diff_confidence
    }
    
    return enhanced_map, debug_info


def frequency_based_diff_fusion(anomaly_map, reconstruction, original,
                               blur_size=21,
                               local_window=15,
                               contrast_multiplier=5):
    """
    使用頻率分析區分結構性差異（低頻）和異常差異（高頻局部）
    
    Args:
        anomaly_map: 原始異常圖 (H, W)
        reconstruction: 重建圖像 (H, W, C) 或 (H, W)
        original: 原始圖像 (H, W, C) 或 (H, W)
        blur_size: 高斯模糊核大小（用於提取低頻）
        local_window: 局部對比度計算窗口大小
        contrast_multiplier: 對比度增強倍數
    
    Returns:
        enhanced_map: 增強後的異常圖
        debug_info: 包含中間結果的字典
    """
    
    # 確保輸入格式一致
    if len(reconstruction.shape) == 2:
        reconstruction = np.expand_dims(reconstruction, axis=2)
    if len(original.shape) == 2:
        original = np.expand_dims(original, axis=2)
    
    diff = np.abs(reconstruction - original)
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)
    
    # 1. 頻率分解
    # 低頻成分 = 結構性差異
    low_freq = cv2.GaussianBlur(diff, (blur_size, blur_size), 0)
    
    # 高頻成分 = 可能的異常
    high_freq = diff - low_freq
    
    # 2. 自適應增強高頻異常
    # 計算局部對比度
    local_max = ndimage.maximum_filter(high_freq, size=local_window)
    local_min = ndimage.minimum_filter(high_freq, size=local_window)
    local_contrast = local_max - local_min
    
    # 高對比度區域更可能是異常
    contrast_weight = np.clip(local_contrast * contrast_multiplier, 0, 1)
    
    # 3. 智能融合
    # 抑制低頻（結構）差異，增強高頻（異常）差異
    processed_diff = low_freq * 0.1 + high_freq * contrast_weight
    
    # 4. 與原始 anomaly map 結合
    # 使用 anomaly map 作為先驗，調整差異的重要性
    prior_weight = np.clip(anomaly_map * 2, 0, 1)
    
    enhanced_map = anomaly_map * 0.7 + processed_diff * 0.3 * prior_weight
    
    enhanced_map = np.clip(enhanced_map, 0, 1)
    
    debug_info = {
        'low_freq': low_freq,
        'high_freq': high_freq,
        'local_contrast': local_contrast,
        'contrast_weight': contrast_weight
    }
    
    return enhanced_map, debug_info


def adaptive_diff_fusion(anomaly_map, reconstruction, original,
                        edge_kernel_size=7,
                        blur_size=15,
                        edge_suppression=0.7,
                        fusion_multiplier=3):
    """
    簡化版：使用局部自適應方法
    
    Args:
        anomaly_map: 原始異常圖 (H, W)
        reconstruction: 重建圖像 (H, W, C) 或 (H, W)
        original: 原始圖像 (H, W, C) 或 (H, W)
        edge_kernel_size: 邊緣擴張核大小
        blur_size: 局部背景估計的模糊核大小
        edge_suppression: 邊緣區域的抑制係數
        fusion_multiplier: 融合權重倍數
    
    Returns:
        enhanced_map: 增強後的異常圖
        debug_info: 包含中間結果的字典
    """
    
    # 確保輸入格式一致
    if len(reconstruction.shape) == 2:
        reconstruction = np.expand_dims(reconstruction, axis=2)
    if len(original.shape) == 2:
        original = np.expand_dims(original, axis=2)
    
    # 1. 計算差異
    diff = np.abs(reconstruction - original)
    if len(diff.shape) == 3:
        diff = diff.mean(axis=2)
    
    # 2. 邊緣抑制
    # 轉換為 uint8 進行邊緣檢測
    if original.max() <= 1:
        original_uint8 = (original * 255).astype(np.uint8)
    else:
        original_uint8 = original.astype(np.uint8)
    
    if len(original_uint8.shape) == 3:
        gray_original = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray_original = original_uint8[:,:,0]
    
    edges = cv2.Canny(gray_original, 50, 150) / 255.0
    edge_mask = cv2.dilate(edges, np.ones((edge_kernel_size, edge_kernel_size)), iterations=1)
    
    # 3. 抑制邊緣區域的差異
    diff_suppressed = diff * (1 - edge_mask * edge_suppression)
    
    # 4. 局部正規化
    # 使差異相對於局部背景更明顯
    blur = cv2.GaussianBlur(diff_suppressed, (blur_size, blur_size), 0)
    diff_normalized = np.where(blur > 0.01, 
                              diff_suppressed / (blur + 0.01), 
                              0)
    
    # 5. 選擇性融合
    # 只在 anomaly map 已經有響應的區域增強
    fusion_weight = np.clip(anomaly_map * fusion_multiplier, 0, 0.3)
    enhanced_map = anomaly_map + diff_normalized * fusion_weight
    
    enhanced_map = np.clip(enhanced_map, 0, 1)
    
    debug_info = {
        'edge_mask': edge_mask,
        'diff_suppressed': diff_suppressed,
        'diff_normalized': diff_normalized,
        'fusion_weight': fusion_weight
    }
    
    return enhanced_map, debug_info


# 融合方法字典，方便選擇
FUSION_METHODS = {
    'intelligent': intelligent_diff_fusion,
    'frequency': frequency_based_diff_fusion,
    'adaptive': adaptive_diff_fusion,
    'none': None  # 不使用融合
}