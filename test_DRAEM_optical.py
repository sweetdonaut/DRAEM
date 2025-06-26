"""
OpticalDataset 專用測試腳本
支援 32-bit TIFF 圖片格式
包含視覺化功能
"""

import torch
import numpy as np
import cv2
import os
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
import tifffile
from datetime import datetime
import re

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def parse_defect_coordinates(filename):
    """從檔名中提取缺陷座標
    
    檔名格式: name#x_y_cur.tiff 或 name#x_y.tiff
    例如: 001#123_456_cur.tiff -> 返回 (123, 456)
    
    Args:
        filename: 檔名字串
        
    Returns:
        (x, y) 座標元組，如果沒有座標則返回 None
    """
    # 使用正則表達式匹配 #x_y 模式
    pattern = r'#(\d+)_(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    
    return None

def visualize_patch(image_path, x, y, model, model_seg, save_dir, patch_size=50, 
                   channels=1, img_height=960, img_width=192, device='cuda:0'):
    """提取並視覺化指定座標的 patch
    
    Args:
        image_path: 圖片路徑
        x, y: 缺陷中心座標
        model: 重建模型
        model_seg: 分割模型
        save_dir: 儲存目錄
        patch_size: patch 大小（預設 50x50）
        channels: 通道數
        img_height, img_width: 模型輸入尺寸
        device: 運算設備
        
    Returns:
        output_path: 儲存的檔案路徑
    """
    # 讀取 TIFF 圖片
    if image_path.endswith('.tiff'):
        image = tifffile.imread(image_path)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 處理通道
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=2)
        
        # 正規化到 0-255 範圍
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        elif image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    else:
        # 處理其他格式
        if channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 計算 patch 邊界（確保不超出圖片範圍）
    half_size = patch_size // 2
    x_start = max(0, x - half_size)
    y_start = max(0, y - half_size)
    x_end = min(image.shape[1], x_start + patch_size)
    y_end = min(image.shape[0], y_start + patch_size)
    
    # 調整起始位置以確保 patch 大小正確
    if x_end - x_start < patch_size:
        x_start = max(0, x_end - patch_size)
    if y_end - y_start < patch_size:
        y_start = max(0, y_end - patch_size)
    
    # 提取原圖 patch
    original_patch = image[y_start:y_end, x_start:x_end]
    
    # 調整整張圖片大小用於模型推理
    image_resized = cv2.resize(image, (img_width, img_height))
    
    # 確保圖片有正確的維度
    if len(image_resized.shape) == 2:
        image_resized = np.expand_dims(image_resized, axis=2)
    
    # 轉換為張量
    image_float = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_float)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        gray_rec = model(image_tensor)
        joined_in = torch.cat((gray_rec, image_tensor), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
    
    # 轉換重建圖為 numpy
    gray_rec = gray_rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if channels == 1:
        gray_rec = (gray_rec[:,:,0] * 255).astype(np.uint8)
    else:
        gray_rec = (gray_rec * 255).astype(np.uint8)
    
    # 將重建圖調整回原始尺寸
    gray_rec_full = cv2.resize(gray_rec, (image.shape[1], image.shape[0]))
    
    # 提取重建圖 patch
    if channels == 1:
        rec_patch = gray_rec_full[y_start:y_end, x_start:x_end]
    else:
        rec_patch = gray_rec_full[y_start:y_end, x_start:x_end]
    
    # 生成熱力圖
    heatmap = out_mask_sm[0, 1].cpu().numpy()
    # 調整熱力圖到原始尺寸
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 提取熱力圖 patch
    heatmap_patch = heatmap[y_start:y_end, x_start:x_end]
    heatmap_patch_colored = cv2.applyColorMap(heatmap_patch, cv2.COLORMAP_JET)
    
    # 為視覺化準備顯示圖片
    if channels == 1:
        original_patch_display = cv2.cvtColor(original_patch[:,:,0], cv2.COLOR_GRAY2RGB)
        rec_patch_display = cv2.cvtColor(rec_patch, cv2.COLOR_GRAY2RGB)
    else:
        original_patch_display = original_patch
        rec_patch_display = rec_patch
    
    # 放大 patch 以便更清楚顯示（放大 3 倍）
    scale_factor = 3
    display_size = patch_size * scale_factor
    
    original_patch_display = cv2.resize(original_patch_display, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    rec_patch_display = cv2.resize(rec_patch_display, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    heatmap_patch_colored = cv2.resize(heatmap_patch_colored, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    
    # 創建包含三個 patch 的大圖（橫向排列，上方留空間給文字）
    text_height = 40  # 文字區域高度（增加一點以配合更大的圖片）
    combined_img = np.zeros((display_size + text_height, display_size * 3, 3), dtype=np.uint8)
    
    # 填充背景色（白色）
    combined_img[:text_height, :] = 255
    
    # 左：原圖 patch
    combined_img[text_height:text_height+display_size, 0:display_size] = original_patch_display
    
    # 中：重建圖 patch
    combined_img[text_height:text_height+display_size, display_size:display_size*2] = rec_patch_display
    
    # 右：熱力圖 patch
    combined_img[text_height:text_height+display_size, display_size*2:display_size*3] = heatmap_patch_colored
    
    # 添加文字標籤
    font_scale = 0.6  # 保持原本的字體大小
    thickness = 1
    
    texts = ["Original", "Reconstructed", "Heatmap"]
    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        text_x = i * display_size + (display_size - text_size[0]) // 2
        text_y = 25
        cv2.putText(combined_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # 儲存結果
    os.makedirs(save_dir, exist_ok=True)
    output_name = os.path.basename(image_path).replace('.tiff', '.png').replace('.jpg', '.png')
    output_path = os.path.join(save_dir, f'vis_{output_name}'.replace('.png', '_patch.png'))
    cv2.imwrite(output_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    return output_path

def visualize_single_image(image_path, model, model_seg, save_dir, channels=1, img_height=960, img_width=192, device='cuda:0', keep_original_size=True):
    """視覺化單張 TIFF 圖片的異常檢測結果"""
    
    # 讀取 TIFF 圖片
    if image_path.endswith('.tiff'):
        image = tifffile.imread(image_path)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 處理通道
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=2)
        
        # 正規化到 0-255 範圍（如果需要）
        if image.dtype == np.float32 or image.dtype == np.float64:
            # 如果是浮點數，檢查值範圍
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                # 正規化到 0-255
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        elif image.max() > 255:
            # 如果是整數但超過 255，也需要正規化
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 為了視覺化，創建顯示用的圖片
        if channels == 1:
            image_display = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
        else:
            image_display = image
    else:
        # 處理其他格式（保留相容性）
        if channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image_display = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_display = image
    
    # 保存原始尺寸和原始顯示圖片
    original_size = image.shape[:2]
    original_display = image_display.copy()
    
    # 調整大小用於模型推理
    image_resized = cv2.resize(image, (img_width, img_height))
    
    # 確保圖片有正確的維度
    if len(image_resized.shape) == 2:
        image_resized = np.expand_dims(image_resized, axis=2)
    
    # 轉換為張量（統一處理為 0-1 範圍）
    image_float = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_float)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        gray_rec = model(image_tensor)
        joined_in = torch.cat((gray_rec, image_tensor), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:,1:,:,:], 7, stride=1, padding=3)
        image_score = torch.max(out_mask_averaged).cpu().item()
    
    # 轉換為 numpy
    gray_rec = gray_rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if channels == 1:
        gray_rec = (gray_rec[:,:,0] * 255).astype(np.uint8)
        rec_display = cv2.cvtColor(gray_rec, cv2.COLOR_GRAY2RGB)
    else:
        rec_display = (gray_rec * 255).astype(np.uint8)
    
    # 生成熱力圖（使用原始 softmax 輸出，與 test_DRAEM.py 一致）
    heatmap = out_mask_sm[0, 1].cpu().numpy()
    # 如果需要調整大小（通常 softmax 輸出已經是正確尺寸）
    if heatmap.shape != (img_height, img_width):
        heatmap = cv2.resize(heatmap, (img_width, img_height))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 決定最終的圖片尺寸
    if keep_original_size and original_size != (img_height, img_width):
        final_height, final_width = original_size
        rec_display = cv2.resize(rec_display, (final_width, final_height))
        heatmap_colored = cv2.resize(heatmap_colored, (final_width, final_height))
        display_img = original_display
    else:
        final_height, final_width = img_height, img_width
        image_display = cv2.resize(image_display, (final_width, final_height))
        display_img = image_display
    
    # 檢查是否有座標，如果有的話準備在圖片上畫 bbox
    filename = os.path.basename(image_path)
    coordinates = parse_defect_coordinates(filename)
    
    # 創建包含三張圖片的大圖（橫向排列，上方留空間給文字）
    text_height = 40  # 文字區域高度
    combined_img = np.zeros((final_height + text_height, final_width * 3, 3), dtype=np.uint8)
    
    # 填充背景色（白色）
    combined_img[:text_height, :] = 255
    
    # 準備要顯示的圖片副本（用於畫 bbox）
    display_img_with_bbox = display_img.copy()
    rec_display_with_bbox = rec_display.copy()
    heatmap_colored_with_bbox = heatmap_colored.copy()
    
    # 如果有座標，在圖片上畫 bbox
    if coordinates is not None:
        x, y = coordinates
        patch_size = 50
        half_size = patch_size // 2
        
        # 計算 bbox 在最終圖片上的位置（需要考慮可能的縮放）
        if keep_original_size and original_size != (img_height, img_width):
            # 座標是原始尺寸的，直接使用
            bbox_x = x
            bbox_y = y
        else:
            # 座標是原始尺寸的，需要縮放到顯示尺寸
            scale_x = final_width / original_size[1]
            scale_y = final_height / original_size[0]
            bbox_x = int(x * scale_x)
            bbox_y = int(y * scale_y)
        
        # 計算 bbox 邊界
        x1 = max(0, bbox_x - half_size)
        y1 = max(0, bbox_y - half_size)
        x2 = min(final_width, x1 + patch_size)
        y2 = min(final_height, y1 + patch_size)
        
        # 調整起始位置以確保 bbox 大小正確
        if x2 - x1 < patch_size:
            x1 = max(0, x2 - patch_size)
        if y2 - y1 < patch_size:
            y1 = max(0, y2 - patch_size)
        
        # 在三張圖片上都畫青色 bbox
        bbox_color = (0, 255, 255)  # 青色 (cyan) in RGB
        bbox_thickness = 2
        
        cv2.rectangle(display_img_with_bbox, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
        cv2.rectangle(rec_display_with_bbox, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
        cv2.rectangle(heatmap_colored_with_bbox, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
    
    # 左：原圖
    combined_img[text_height:final_height+text_height, 0:final_width] = display_img_with_bbox
    
    # 中：重建圖
    combined_img[text_height:final_height+text_height, final_width:final_width*2] = rec_display_with_bbox
    
    # 右：熱力圖
    combined_img[text_height:final_height+text_height, final_width*2:final_width*3] = heatmap_colored_with_bbox
    
    # 添加文字標籤（黑色文字，使用較清晰的字型）
    # 根據圖片寬度動態調整字體大小，但限制最大值
    font_scale = min(2.0, max(0.8, final_width / 500))  # 調整比例，限制最大為 2.0
    thickness = max(2, int(font_scale * 1.5))  # 增加線條粗細
    
    # 計算文字位置（置中）
    texts = ["Original", "Reconstructed", "Heatmap"]
    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        text_x = i * final_width + (final_width - text_size[0]) // 2
        text_y = 30
        cv2.putText(combined_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness)
    
    # 儲存結果
    os.makedirs(save_dir, exist_ok=True)
    output_name = os.path.basename(image_path).replace('.tiff', '.png').replace('.jpg', '.png')
    output_path = os.path.join(save_dir, f'vis_{output_name}')
    cv2.imwrite(output_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    # 檢查檔名是否包含座標，如果有的話生成 patch 視覺化
    # （已經在上面解析過座標了，所以直接使用）
    patch_output_path = None
    
    if coordinates is not None:
        x, y = coordinates
        patch_output_path = visualize_patch(
            image_path, x, y, model, model_seg, save_dir,
            patch_size=50, channels=channels, img_height=img_height, 
            img_width=img_width, device=device
        )
        print(f"  生成 patch 視覺化: {patch_output_path}")
    
    return image_score, output_path

def test_on_device(obj_names, args, model_name, device):
    """測試模型效能"""
    
    # 載入模型
    checkpoint_path = os.path.join(args.checkpoint_path, model_name + ".pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 從檢查點獲取模型參數
    channels = checkpoint.get('channels', 1)
    img_height = checkpoint.get('img_height', 960)
    img_width = checkpoint.get('img_width', 192)
    
    print(f"模型資訊:")
    print(f"- 通道數: {channels}")
    print(f"- 圖片尺寸: {img_height}x{img_width}")
    
    # 初始化模型
    model = ReconstructiveSubNetwork(in_channels=channels, out_channels=channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    model_seg = DiscriminativeSubNetwork(in_channels=channels*2, out_channels=2).to(device)
    seg_checkpoint_path = os.path.join(args.checkpoint_path, model_name + "_seg.pth")
    model_seg.load_state_dict(torch.load(seg_checkpoint_path, map_location=device))
    model_seg.eval()
    
    # 測試每個類別
    results = {}
    
    for obj_name in obj_names:
        print(f"\n測試類別: {obj_name}")
        
        # 準備資料載入器
        dataset = MVTecDRAEMTestDataset(
            root_dir=os.path.join(args.data_path, 'test'),
            resize_shape=[img_height, img_width],
            channels=channels
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        total_pixel_scores = []
        total_gt_pixel_scores = []
        anomaly_score_gt = []
        anomaly_score_prediction = []
        
        # 創建輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./outputs/test_results", model_name, obj_name, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # 處理每張圖片
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].to(device)
            is_normal = sample_batched["has_anomaly"][0].cpu().numpy()[0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.cpu().numpy()[0, :, :, :].transpose((1, 2, 0))
            
            # 推理
            with torch.no_grad():
                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec, gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                
                if args.save_images and i_batch < 10:  # 只儲存前10張
                    # 視覺化
                    image_path = dataset.images[i_batch]
                    score, vis_path = visualize_single_image(
                        image_path, model, model_seg, output_dir,
                        channels=channels, img_height=img_height, img_width=img_width, device=device,
                        keep_original_size=True
                    )
                    print(f"  圖片 {i_batch+1}: 異常分數 = {score:.4f}, 結果儲存至: {vis_path}")
                
                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:,1:,:,:], 21, stride=1, padding=21//2)
                image_score = torch.max(out_mask_averaged).cpu().item()
                anomaly_score_prediction.append(image_score)
                
                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_averaged.cpu().numpy().flatten()
                total_pixel_scores.extend(flat_out_mask)
                total_gt_pixel_scores.extend(flat_true_mask)
        
        # 計算 AUC
        if len(np.unique(anomaly_score_gt)) > 1:
            auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
            print(f"  圖片級別 AUROC: {auroc:.4f}")
        else:
            auroc = None
            print(f"  圖片級別 AUROC: N/A (只有一個類別)")
        
        if len(np.unique(total_gt_pixel_scores)) > 1:
            pixel_auroc = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
            print(f"  像素級別 AUROC: {pixel_auroc:.4f}")
        else:
            pixel_auroc = None
            print(f"  像素級別 AUROC: N/A (沒有異常像素)")
        
        results[obj_name] = {
            'image_auroc': auroc,
            'pixel_auroc': pixel_auroc,
            'output_dir': output_dir
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DRAEM OpticalDataset 測試腳本')
    
    # 基本參數
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--data_path', type=str, default='./OpticalDataset', 
                        help='Path to OpticalDataset')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Path to checkpoints directory')
    
    # 模型相關
    parser.add_argument('--model_name', type=str, required=False,
                        help='Model name without .pth extension. If not provided, uses the latest model.')
    
    # 測試選項
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--image_path', type=str, required=False,
                        help='Path to single image for visualization')
    parser.add_argument('--output_dir', type=str, default='./outputs/test_results',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # 設定裝置
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
        print("警告：CUDA 不可用，使用 CPU")
    
    # 如果沒有指定模型，使用最新的 optical 模型
    if args.model_name is None:
        import glob
        optical_models = glob.glob(os.path.join(args.checkpoint_path, "DRAEM_optical_*.pth"))
        # 過濾掉 _seg 檔案
        optical_models = [m for m in optical_models if not m.endswith("_seg.pth")]
        if not optical_models:
            print("錯誤：找不到任何 OpticalDataset 模型")
            print("請先使用 train_DRAEM_optical.py 訓練模型")
            return
        
        # 選擇最新的模型
        args.model_name = os.path.basename(max(optical_models, key=os.path.getctime))[:-4]
        print(f"使用最新模型: {args.model_name}")
    
    # 單張圖片模式
    if args.image_path:
        print(f"\n處理單張圖片: {args.image_path}")
        
        # 載入模型
        checkpoint_path = os.path.join(args.checkpoint_path, args.model_name + ".pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        channels = checkpoint.get('channels', 1)
        img_height = checkpoint.get('img_height', 960)
        img_width = checkpoint.get('img_width', 192)
        
        model = ReconstructiveSubNetwork(in_channels=channels, out_channels=channels).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        model_seg = DiscriminativeSubNetwork(in_channels=channels*2, out_channels=2).to(device)
        seg_checkpoint_path = os.path.join(args.checkpoint_path, args.model_name + "_seg.pth")
        model_seg.load_state_dict(torch.load(seg_checkpoint_path, map_location=device))
        model_seg.eval()
        
        # 視覺化
        score, output_path = visualize_single_image(
            args.image_path, model, model_seg, args.output_dir,
            channels=channels, img_height=img_height, img_width=img_width, device=device,
            keep_original_size=True
        )
        
        print(f"異常分數: {score:.4f}")
        print(f"結果儲存至: {output_path}")
    
    else:
        # 完整測試模式
        print(f"\n開始測試 OpticalDataset")
        print(f"模型: {args.model_name}")
        print(f"資料路徑: {args.data_path}")
        
        # 獲取所有測試類別
        test_dir = os.path.join(args.data_path, 'test')
        if not os.path.exists(test_dir):
            print(f"錯誤：找不到測試目錄 {test_dir}")
            return
        
        obj_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        print(f"找到類別: {obj_names}")
        
        # 執行測試
        results = test_on_device(obj_names, args, args.model_name, device)
        
        # 顯示總結
        print(f"\n{'='*60}")
        print("測試結果總結")
        print(f"{'='*60}")
        for obj_name, result in results.items():
            print(f"\n{obj_name}:")
            if result['image_auroc'] is not None:
                print(f"  圖片級別 AUROC: {result['image_auroc']:.4f}")
            if result['pixel_auroc'] is not None:
                print(f"  像素級別 AUROC: {result['pixel_auroc']:.4f}")
            print(f"  輸出目錄: {result['output_dir']}")

if __name__ == "__main__":
    main()