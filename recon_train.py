import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unet import ReconstructiveSubNetwork, ReconstructiveSubNetworkWithSkip, ReconstructiveVAE
from loss import SSIM
import os
import argparse


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    dtd_path = args.dtd_path
    if dtd_path is not None and not os.path.exists(dtd_path):
        print(f"DTD path not found: {dtd_path}, using random noise instead.")
        dtd_path = None

    bw_str = f'_bw{args.base_width}' if args.base_width != 128 else ''

    if args.model_type == 'skip':
        skip_str = ''.join(map(str, sorted(args.skip_layers)))
        run_name = f'recon_skip{skip_str}_{args.loss_type}_lr{args.lr}_ep{args.epochs}_bs{args.bs}_ch{args.channels}_{args.img_size[0]}x{args.img_size[1]}{bw_str}'
        model = ReconstructiveSubNetworkWithSkip(in_channels=args.channels, out_channels=args.channels, base_width=args.base_width, skip_layers=args.skip_layers)
    elif args.model_type == 'vae':
        run_name = f'recon_vae_{args.loss_type}_lr{args.lr}_ep{args.epochs}_bs{args.bs}_ch{args.channels}_{args.img_size[0]}x{args.img_size[1]}{bw_str}'
        model = ReconstructiveVAE(in_channels=args.channels, out_channels=args.channels, base_width=args.base_width)
    else:
        run_name = f'recon_original_{args.loss_type}_lr{args.lr}_ep{args.epochs}_bs{args.bs}_ch{args.channels}_{args.img_size[0]}x{args.img_size[1]}{bw_str}'
        model = ReconstructiveSubNetwork(in_channels=args.channels, out_channels=args.channels, base_width=args.base_width)

    model.cuda()
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs*0.8), int(args.epochs*0.9)], gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()

    dataset = MVTecDRAEMTrainDataset(
        args.data_path + "/train/good/",
        dtd_path,
        resize_shape=args.img_size,
        channels=args.channels
    )
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    model_info = f"Model type: {args.model_type}"
    if args.model_type == 'skip':
        model_info += f" (skip_layers={args.skip_layers})"
    if args.base_width != 128:
        model_info += f" (base_width={args.base_width})"
    print(model_info)
    print(f"Loss type: {args.loss_type}")
    print(f"DTD: {dtd_path if dtd_path else 'None (using random noise)'}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Image size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"Channels: {args.channels}")
    print(f"Run name: {run_name}")
    print("-" * 50)

    num_batches = len(dataloader)

    for epoch in range(args.epochs):
        epoch_loss = 0

        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()

            if args.model_type == 'vae':
                gray_rec, mu, logvar = model(aug_gray_batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            else:
                gray_rec = model(aug_gray_batch)
                kl_loss = 0

            if args.loss_type == 'l2':
                recon_loss = loss_l2(gray_rec, gray_batch)
            elif args.loss_type == 'ssim':
                recon_loss = loss_ssim(gray_rec, gray_batch)
            else:  # l2+ssim
                recon_loss = loss_l2(gray_rec, gray_batch) + loss_ssim(gray_rec, gray_batch)

            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i_batch % 10 == 0 or i_batch == num_batches - 1:
                current_lr = get_lr(optimizer)
                progress = (i_batch + 1) / num_batches * 100
                if args.model_type == 'vae':
                    print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                          f'Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}) - LR: {current_lr:.6f}', end='', flush=True)
                else:
                    print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                          f'Loss: {loss.item():.4f} - LR: {current_lr:.6f}', end='', flush=True)

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        print(f'\nEpoch [{epoch+1}/{args.epochs}] Summary - Avg Loss: {avg_loss:.4f}')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': args.model_type,
            'skip_layers': args.skip_layers if args.model_type == 'skip' else None,
            'base_width': args.base_width,
            'loss_type': args.loss_type,
            'channels': args.channels,
            'img_height': args.img_size[0],
            'img_width': args.img_size[1],
            'epoch': epoch + 1,
            'loss': avg_loss
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, run_name + ".pth"))

    print("-" * 50)
    print(f"Training complete. Model saved to: {os.path.join(args.checkpoint_path, run_name + '.pth')}")


def main():
    parser = argparse.ArgumentParser(description='Train Reconstructive Network')
    parser.add_argument('--model_type', type=str, default='original', choices=['original', 'skip', 'vae'],
                        help='Model type: original, skip, or vae')
    parser.add_argument('--skip_layers', type=int, nargs='+', default=[4],
                        help='Skip layers for skip model: 2, 3, 4 or combinations like 2 3 4')
    parser.add_argument('--base_width', type=int, default=128,
                        help='Base channel width for the network')
    parser.add_argument('--loss_type', type=str, default='l2+ssim', choices=['l2', 'ssim', 'l2+ssim'],
                        help='Loss type: l2, ssim, or l2+ssim')
    parser.add_argument('--bs', type=int, required=True, help='Batch size')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dtd_path', type=str, default=None,
                        help='Path to DTD dataset. If not provided or not exists, uses random noise.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Image size [height, width]')

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)


if __name__ == "__main__":
    main()
