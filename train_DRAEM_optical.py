"""
OpticalDataset 專用訓練腳本
支援 32-bit TIFF 圖片格式
針對非正方形圖片（960x192）優化
"""

import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
import time

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

def train_on_device(obj_names, args):
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    for obj_name in obj_names:
        # 修改命名規則以區分 Optical 模型
        run_name = f'DRAEM_optical_{args.lr}_{args.epochs}_bs{args.bs}_ch{args.channels}_{args.img_size[0]}x{args.img_size[1]}'
        
        print(f"\n{'='*60}")
        print(f"訓練 OpticalDataset 模型")
        print(f"模型名稱: {run_name}")
        print(f"圖片尺寸: {args.img_size[0]}x{args.img_size[1]}")
        print(f"通道數: {args.channels}")
        print(f"Batch size: {args.bs}")
        print(f"{'='*60}\n")

        # 初始化模型
        model = ReconstructiveSubNetwork(in_channels=args.channels, out_channels=args.channels)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=args.channels*2, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        # 優化器設定
        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        # 損失函數
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        # 資料載入器 - 針對 OpticalDataset 優化
        dataset = MVTecDRAEMTrainDataset(
            args.data_path + "/train/good/", 
            args.anomaly_source_path, 
            resize_shape=args.img_size, 
            channels=args.channels
        )
        
        # 使用較少的 workers 以避免記憶體問題
        dataloader = DataLoader(
            dataset, 
            batch_size=args.bs,
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        print(f"資料集大小: {len(dataset)}")
        print(f"Batch 數量: {len(dataloader)}")

        n_iter = 0
        num_batches = len(dataloader)
        
        # 訓練迴圈
        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_losses = {'l2': 0, 'ssim': 0, 'focal': 0, 'total': 0}
            batch_times = []
            
            for i_batch, sample_batched in enumerate(dataloader):
                batch_start = time.time()
                
                # 資料載入
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                # 前向傳播
                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # 計算損失
                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新統計資料
                epoch_losses['l2'] += l2_loss.item()
                epoch_losses['ssim'] += ssim_loss.item()
                epoch_losses['focal'] += segment_loss.item()
                epoch_losses['total'] += loss.item()
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # 顯示訓練進度 (每個 batch 都顯示，因為 OpticalDataset 較慢)
                if i_batch % 1 == 0:
                    current_lr = get_lr(optimizer)
                    progress = (i_batch + 1) / num_batches * 100
                    avg_batch_time = sum(batch_times) / len(batch_times)
                    eta = avg_batch_time * (num_batches - i_batch - 1)
                    
                    print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                          f'L2: {l2_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, Focal: {segment_loss.item():.4f}, '
                          f'Total: {loss.item():.4f} - LR: {current_lr:.6f} - '
                          f'Batch time: {batch_time:.2f}s - ETA: {eta:.0f}s', end='', flush=True)

                n_iter +=1

            scheduler.step()
            
            # Epoch 總結
            epoch_time = time.time() - epoch_start
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f'\nEpoch [{epoch+1}/{args.epochs}] 完成 - 耗時: {epoch_time:.1f}s')
            print(f'平均損失 - L2: {avg_losses["l2"]:.4f}, SSIM: {avg_losses["ssim"]:.4f}, '
                  f'Focal: {avg_losses["focal"]:.4f}, Total: {avg_losses["total"]:.4f}')
            print(f'平均 batch 時間: {sum(batch_times)/len(batch_times):.2f}s')

            # 儲存模型檢查點
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'channels': args.channels,
                'img_height': args.img_size[0],
                'img_width': args.img_size[1],
                'dataset_type': 'optical',
                'epoch': epoch + 1
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, run_name+".pth"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pth"))
            print(f'模型已儲存: {run_name}.pth')

        print(f"\n{'='*60}")
        print(f"訓練完成！")
        print(f"最終模型: {run_name}.pth")
        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DRAEM OpticalDataset 訓練腳本')
    parser.add_argument('--bs', action='store', type=int, required=True, help='Batch size')
    parser.add_argument('--lr', action='store', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs')
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False, help='GPU ID')
    parser.add_argument('--data_path', action='store', type=str, default='./OpticalDataset', 
                        help='Path to OpticalDataset. Default: ./OpticalDataset')
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=False, default=None,
                        help='Path to anomaly source images. If not provided, random noise will be used.')
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoints',
                        help='Path to save checkpoints. Default: ./checkpoints')
    parser.add_argument('--channels', action='store', type=int, default=1, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB). Default: 1')
    parser.add_argument('--img_size', action='store', type=int, nargs=2, default=[960, 192],
                        help='Image size for training as [height, width]. Default: [960, 192]')
    parser.add_argument('--num_workers', action='store', type=int, default=4,
                        help='Number of data loading workers. Default: 4')

    args = parser.parse_args()

    # 建議檢查 batch size 以避免 GPU 記憶體溢出
    if args.bs > 4 and args.img_size[0] * args.img_size[1] > 100000:
        print(f"警告：大尺寸圖片 ({args.img_size[0]}x{args.img_size[1]}) 配合 batch size {args.bs} 可能導致 GPU 記憶體不足！")
        print("建議使用較小的 batch size (如 2 或 3)")

    with torch.cuda.device(args.gpu_id):
        train_on_device(['Optical'], args)

if __name__=="__main__":
    main()