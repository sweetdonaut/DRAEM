import torch
import numpy as np
import cv2
import os
import argparse
import glob
import tifffile
from model_unet import ReconstructiveSubNetwork, ReconstructiveSubNetworkWithSkip, ReconstructiveVAE


def load_model(model_path, device='cuda:0'):
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get('model_type', 'original')
    skip_layers = checkpoint.get('skip_layers', [4])
    base_width = checkpoint.get('base_width', 128)
    channels = checkpoint.get('channels', 3)
    img_height = checkpoint.get('img_height', 256)
    img_width = checkpoint.get('img_width', 256)
    loss_type = checkpoint.get('loss_type', 'unknown')

    model_info = f"Model type: {model_type}"
    if model_type == 'skip':
        model_info += f" (skip_layers={skip_layers})"
    if base_width != 128:
        model_info += f" (base_width={base_width})"
    print(model_info)
    print(f"Loss type: {loss_type}")
    print(f"Channels: {channels}")
    print(f"Image size: {img_height}x{img_width}")

    if model_type == 'skip':
        model = ReconstructiveSubNetworkWithSkip(in_channels=channels, out_channels=channels, base_width=base_width, skip_layers=skip_layers)
    elif model_type == 'vae':
        model = ReconstructiveVAE(in_channels=channels, out_channels=channels, base_width=base_width)
    else:
        model = ReconstructiveSubNetwork(in_channels=channels, out_channels=channels, base_width=base_width)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_type, channels, img_height, img_width


def load_image(image_path, channels, img_height, img_width):
    if image_path.endswith('.tiff'):
        image = tifffile.imread(image_path)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=2)
    else:
        if channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
        else:
            image = cv2.imread(image_path)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (img_width, img_height))
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    return image


def visualize_single_image(image_path, model, model_type, channels, img_height, img_width, output_dir, device='cuda:0'):
    image = load_image(image_path, channels, img_height, img_width)

    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == 'vae':
            reconstructed = model.reconstruct(image_tensor)
        else:
            reconstructed = model(image_tensor)

    reconstructed_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)

    diff = np.abs(image_normalized - reconstructed_np)
    if channels == 1:
        diff_gray = diff[:, :, 0]
    else:
        diff_gray = np.mean(diff, axis=2)

    diff_normalized = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)

    if channels == 1:
        original_display = cv2.cvtColor((image_normalized[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        reconstructed_display = cv2.cvtColor((reconstructed_np[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        original_display = (image_normalized * 255).astype(np.uint8)
        reconstructed_display = (reconstructed_np * 255).astype(np.uint8)

    diff_heatmap = cv2.applyColorMap((diff_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)

    text_height = 40
    h, w = img_height, img_width
    combined = np.zeros((h + text_height, w * 3, 3), dtype=np.uint8)
    combined[:text_height, :] = 255

    combined[text_height:, 0:w] = original_display
    combined[text_height:, w:w*2] = reconstructed_display
    combined[text_height:, w*2:w*3] = diff_heatmap

    titles = ["Original", "Reconstructed", "Difference"]
    for i, title in enumerate(titles):
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
        text_x = i * w + (w - text_size[0]) // 2
        cv2.putText(combined, title, (text_x, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(output_dir, f'vis_{name}.png')
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    mse = np.mean(diff ** 2)
    print(f"{filename} - MSE: {mse:.6f} - Saved: {save_path}")

    return mse


def main():
    parser = argparse.ArgumentParser(description='Visualize Reconstructive Network Results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, default=None, help='Path to single image')
    parser.add_argument('--test_dir', type=str, default=None, help='Path to test directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/recon_vis/', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--max_images', type=int, default=0, help='Max images to process (0 = all)')

    args = parser.parse_args()

    if args.image_path is None and args.test_dir is None:
        print("Please provide --image_path or --test_dir")
        return

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    model, model_type, channels, img_height, img_width = load_model(args.model_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_path:
        visualize_single_image(args.image_path, model, model_type, channels, img_height, img_width, args.output_dir, device)
    elif args.test_dir:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(args.test_dir, '**', ext), recursive=True))

        print(f"Found {len(image_files)} images")
        print("-" * 50)

        mse_scores = []
        files_to_process = image_files if args.max_images == 0 else image_files[:args.max_images]
        for img_path in files_to_process:
            mse = visualize_single_image(img_path, model, model_type, channels, img_height, img_width, args.output_dir, device)
            mse_scores.append(mse)

        print("-" * 50)
        print(f"Processed {len(mse_scores)} images")
        print(f"Average MSE: {np.mean(mse_scores):.6f}")
        print(f"Max MSE: {np.max(mse_scores):.6f}")
        print(f"Min MSE: {np.min(mse_scores):.6f}")


if __name__ == "__main__":
    main()
