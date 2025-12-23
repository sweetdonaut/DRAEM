import argparse
import os
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import InpaintingTestDataset
from model_unet import UNet, FFCUNet


def draw_roi_bbox(image, roi_path, scale_x, scale_y, x_len=10):
    img = image.copy()
    roi_df = pd.read_csv(roi_path)
    for _, row in roi_df.iterrows():
        cx = int(row['raw_x'] * scale_x)
        cy = int(row['raw_y'] * scale_y)
        half_x = int(x_len * scale_x) // 2
        half_y = int(row['y_len'] * scale_y) // 2
        x1, y1 = cx - half_x, cy - half_y
        x2, y2 = cx + half_x, cy + half_y
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model_type == "ffc":
        model = FFCUNet(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    else:
        model = UNet(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    dataset = InpaintingTestDataset(
        root_dir=args.data_path,
        defect_type=args.defect_type,
        resize_shape=(args.img_size, args.img_size),
        dilate_size=args.dilate_size,
        border_margin=args.border_margin
    )
    print(f"Test dataset size: {len(dataset)}")

    output_dir = os.path.join(args.output_path, args.defect_type)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            input_img = sample['input'].unsqueeze(0).to(device)
            target_img = sample['target'].numpy().transpose(1, 2, 0)
            mask = sample['mask'].numpy().squeeze()
            img_path = sample['path']

            output = model(input_img)
            output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            recon_blend = target_img * (1 - mask[:, :, None]) + output_img * mask[:, :, None]
            diff = np.abs(target_img - output_img) * mask[:, :, None]
            diff_gray = np.mean(diff, axis=2)

            original_vis = (target_img * 255).astype(np.uint8)
            recon_vis = (recon_blend * 255).astype(np.uint8)

            filename = os.path.splitext(os.path.basename(img_path))[0]
            roi_path = os.path.join(args.data_path, f"test_roi/{args.defect_type}/{filename}.csv")

            original_img = cv2.imread(img_path)
            scale_x = args.img_size / original_img.shape[1]
            scale_y = args.img_size / original_img.shape[0]
            recon_with_bbox = draw_roi_bbox(recon_vis, roi_path, scale_x, scale_y)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(cv2.cvtColor(recon_with_bbox, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")

            axes[2].imshow(diff_gray, cmap="hot", vmin=0, vmax=args.vmax)
            axes[2].set_title("Difference")
            axes[2].axis("off")

            plt.subplots_adjust(wspace=0.02, hspace=0)
            plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=100, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            print(f"Saved: {filename}.png")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets/carpet_gray")
    parser.add_argument("--model_path", type=str, default="checkpoints/model_final.pth")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--defect_type", type=str, default="metal_contamination")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "ffc"])
    parser.add_argument("--dilate_size", type=int, default=0)
    parser.add_argument("--border_margin", type=int, default=0)
    parser.add_argument("--vmax", type=float, default=0.1)
    args = parser.parse_args()

    test(args)
