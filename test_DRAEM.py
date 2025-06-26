"""
用於單獨視覺化測試結果的腳本
可以在訓練完成後單獨運行來生成視覺化結果
"""

import torch
import numpy as np
import cv2
import os
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import argparse

def visualize_single_image(image_path, model, model_seg, save_dir, channels=3, img_height=256, img_width=256, device='cuda:0'):
    """視覺化單張圖片的異常檢測結果"""
    
    # 讀取圖片
    if channels == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        # 為了視覺化，將灰階圖轉為3通道
        image_display = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.imread(image_path)
        # 如果是灰階圖，自動轉為RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_display = image
    
    original_size = image.shape[:2]
    
    # 調整大小
    if channels == 1:
        image = cv2.resize(image[:,:,0], (img_width, img_height))
        image = np.expand_dims(image, axis=2)
    else:
        image = cv2.resize(image, (img_width, img_height))
    image_display = cv2.resize(image_display, (img_width, img_height))
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        gray_rec = model(image_tensor)
        joined_in = torch.cat((gray_rec, image_tensor), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        
    # 處理結果
    heatmap = out_mask_sm[0, 1].cpu().numpy()
    out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 7, stride=1, padding=3).cpu().numpy()
    anomaly_score = np.max(out_mask_averaged)
    
    # 準備視覺化
    original_img = image_display
    reconstructed_img = (gray_rec[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # 如果是單通道，轉換重建圖為3通道以便顯示
    if channels == 1:
        reconstructed_img = cv2.cvtColor(reconstructed_img[:,:,0], cv2.COLOR_GRAY2RGB)
    
    # 創建熱力圖
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 創建包含三張圖片的大圖（橫向排列，上方留空間給文字）
    h, w = img_height, img_width
    text_height = 40  # 文字區域高度
    combined_img = np.zeros((h + text_height, w*3, 3), dtype=np.uint8)
    
    # 填充背景色（白色）
    combined_img[:text_height, :] = 255
    
    # 左：原圖
    combined_img[text_height:h+text_height, 0:w] = original_img
    
    # 中：重建圖
    combined_img[text_height:h+text_height, w:w*2] = reconstructed_img
    
    # 右：純熱力圖
    combined_img[text_height:h+text_height, w*2:w*3] = heatmap_colored
    
    # 添加文字標籤（黑色文字，使用較清晰的字型）
    # 使用抗鋸齒線條類型
    font_scale = 0.7
    thickness = 1
    
    # 計算文字位置（置中）
    texts = ["Original", "Reconstructed", "Heatmap"]
    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        text_x = i * w + (w - text_size[0]) // 2
        text_y = 28
        cv2.putText(combined_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # 儲存圖片
    filename = os.path.basename(image_path)
    # 直接使用提供的 save_dir，不創建新的子目錄
    save_path = os.path.join(save_dir, f'vis_{filename}')
    cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    print(f"Saved visualization to: {save_path}")
    print(f"Anomaly score: {anomaly_score:.3f}")
    
    return anomaly_score

def main():
    parser = argparse.ArgumentParser(description='Visualize DRAEM results')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Path to checkpoint directory')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (without extension)')
    parser.add_argument('--image_path', type=str, help='Path to single image to visualize')
    parser.add_argument('--test_dir', type=str, help='Path to test directory with multiple images')
    parser.add_argument('--output_dir', type=str, default='./outputs/visualizations/', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--channels', type=int, default=None, choices=[1, 3],
                        help='Number of input channels. If not specified, will try to detect from model.')
    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                        help='Image size for testing as [height, width]. If not specified, will try to detect from model.')
    
    args = parser.parse_args()
    
    # 設定設備
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # 載入模型並檢查通道數
    checkpoint_path = os.path.join(args.checkpoint_path, args.model_name + ".pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 檢測通道數
    if args.channels is None:
        if isinstance(checkpoint, dict) and 'channels' in checkpoint:
            channels = checkpoint['channels']
            print(f"Detected channels from model: {channels}")
        else:
            # 舊模型默認為3通道
            channels = 3
            print("Using default channels: 3 (legacy model)")
    else:
        channels = args.channels
        print(f"Using specified channels: {channels}")
    
    # 檢測圖片尺寸
    if args.img_size is None:
        if isinstance(checkpoint, dict) and 'img_height' in checkpoint and 'img_width' in checkpoint:
            img_height = checkpoint['img_height']
            img_width = checkpoint['img_width']
            print(f"Detected image size from model: {img_height}x{img_width}")
        elif isinstance(checkpoint, dict) and 'img_size' in checkpoint:
            # 相容舊格式（單一數值）
            img_height = img_width = checkpoint['img_size']
            print(f"Detected image size from model: {img_height}x{img_width} (legacy square format)")
        else:
            # 舊模型默認為256x256
            img_height = img_width = 256
            print("Using default image size: 256x256 (legacy model)")
    else:
        img_height, img_width = args.img_size
        print(f"Using specified image size: {img_height}x{img_width}")
    
    # 載入模型
    model = ReconstructiveSubNetwork(in_channels=channels, out_channels=channels)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 兼容舊格式
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    model_seg = DiscriminativeSubNetwork(in_channels=channels*2, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.model_name + "_seg.pth"), 
                                        map_location=device))
    model_seg.to(device)
    model_seg.eval()
    
    # 使用指定的輸出目錄，不創建新的時間戳子目錄
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if args.image_path:
        # 處理單張圖片
        visualize_single_image(args.image_path, model, model_seg, save_dir, channels, img_height, img_width, device)
    elif args.test_dir:
        # 處理目錄中的所有圖片
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(args.test_dir, '**', ext), recursive=True))
        
        print(f"Found {len(image_files)} images")
        
        scores = []
        for img_path in image_files[:20]:  # 最多處理20張
            score = visualize_single_image(img_path, model, model_seg, save_dir, channels, img_height, img_width, device)
            scores.append(score)
        
        print(f"\nProcessed {len(scores)} images")
        print(f"Average anomaly score: {np.mean(scores):.3f}")
        print(f"Max anomaly score: {np.max(scores):.3f}")
        print(f"Min anomaly score: {np.min(scores):.3f}")
    else:
        print("Please provide either --image_path or --test_dir")

if __name__ == "__main__":
    import glob
    main()