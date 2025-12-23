import argparse
import torch
import torch.optim as optim
from data_loader import InpaintingTrainDataset
from model_unet import UNet, FFCUNet
from loss import L2Loss, SSIMLoss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = InpaintingTrainDataset(
        root_dir=args.data_path,
        resize_shape=(args.img_size, args.img_size),
        dilate_size=args.dilate_size,
        border_margin=args.border_margin
    )
    dataloader = dataset.get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    print(f"Dataset size: {len(dataset)}")

    if args.model_type == "ffc":
        model = FFCUNet(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    else:
        model = UNet(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    print(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    l2_loss_fn = L2Loss()
    ssim_loss_fn = SSIMLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_img = batch['input'].to(device)
            target_img = batch['target'].to(device)

            optimizer.zero_grad()
            mask = batch['mask'].to(device)
            output = model(input_img)

            output_roi = output * mask
            target_roi = target_img * mask
            l2 = l2_loss_fn(output_roi, target_roi)
            ssim = ssim_loss_fn(output_roi, target_roi)

            if args.loss_type == "ssim":
                loss = ssim
            elif args.loss_type == "l2":
                loss = l2
            else:
                loss = l2 + ssim

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (L2: {l2.item():.4f}, SSIM: {ssim.item():.4f})")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.save_path}/model_final.pth")
            print(f"Model saved to {args.save_path}/model_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="datasets/carpet_gray")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="unet", choices=["unet", "ffc"])
    parser.add_argument("--loss_type", type=str, default="both", choices=["l2", "ssim", "both"])
    parser.add_argument("--dilate_size", type=int, default=0)
    parser.add_argument("--border_margin", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    import os
    os.makedirs(args.save_path, exist_ok=True)

    train(args)
