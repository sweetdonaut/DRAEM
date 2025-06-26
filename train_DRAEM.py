#import disable_debugger  # 禁用所有偵錯器
import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os

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
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_ch"+str(args.channels)+"_"+str(args.img_size[0])+"x"+str(args.img_size[1])+"_RSEM_"


        model = ReconstructiveSubNetwork(in_channels=args.channels, out_channels=args.channels)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=args.channels*2, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + "/train/good/", args.anomaly_source_path, 
                                        resize_shape=args.img_size, channels=args.channels)
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=8)
        print("Dataset size:", len(dataset))

        n_iter = 0
        num_batches = len(dataloader)
        
        for epoch in range(args.epochs):
            epoch_losses = {'l2': 0, 'ssim': 0, 'focal': 0, 'total': 0}
            
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                # Update epoch losses
                epoch_losses['l2'] += l2_loss.item()
                epoch_losses['ssim'] += ssim_loss.item()
                epoch_losses['focal'] += segment_loss.item()
                epoch_losses['total'] += loss.item()

                # Print training progress
                if i_batch % 10 == 0 or i_batch == num_batches - 1:  # Print every 10 batches or last batch
                    current_lr = get_lr(optimizer)
                    progress = (i_batch + 1) / num_batches * 100
                    print(f'\rEpoch [{epoch+1}/{args.epochs}] - Batch [{i_batch+1}/{num_batches}] ({progress:.1f}%) - '
                          f'L2: {l2_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, Focal: {segment_loss.item():.4f}, '
                          f'Total: {loss.item():.4f} - LR: {current_lr:.6f}', end='', flush=True)

                n_iter +=1

            scheduler.step()
            
            # Print epoch summary
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f'\nEpoch [{epoch+1}/{args.epochs}] Summary - '
                  f'Avg L2: {avg_losses["l2"]:.4f}, Avg SSIM: {avg_losses["ssim"]:.4f}, '
                  f'Avg Focal: {avg_losses["focal"]:.4f}, Avg Total: {avg_losses["total"]:.4f}')

            # 儲存模型和通道資訊
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'channels': args.channels,
                'img_height': args.img_size[0],
                'img_width': args.img_size[1]
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, run_name+".pth"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pth"))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=False, default=None,
                        help='Path to anomaly source images (e.g., DTD dataset). If not provided, random noise will be used.')
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--channels', action='store', type=int, default=3, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB). Default: 3')
    parser.add_argument('--img_size', action='store', type=int, nargs=2, default=[256, 256],
                        help='Image size for training as [height, width]. Default: [256, 256]')

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(['RSEM'], args)

if __name__=="__main__":
    main()

