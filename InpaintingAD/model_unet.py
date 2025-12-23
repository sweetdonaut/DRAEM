import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.shape
        fft = torch.fft.rfft2(x, norm='ortho')
        fft = torch.stack([fft.real, fft.imag], dim=-1)
        fft = fft.permute(0, 1, 4, 2, 3).contiguous()
        fft = fft.view(batch, -1, h, w // 2 + 1)

        fft = self.conv(fft)
        fft = self.bn(fft)
        fft = self.relu(fft)

        fft = fft.view(batch, -1, 2, h, w // 2 + 1)
        fft = fft.permute(0, 1, 3, 4, 2).contiguous()
        fft = torch.complex(fft[..., 0], fft[..., 1])
        out = torch.fft.irfft2(fft, s=(h, w), norm='ortho')
        return out


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_global=0.5):
        super(FFC, self).__init__()
        self.ratio_global = ratio_global
        in_local = int(in_channels * (1 - ratio_global))
        in_global = in_channels - in_local
        out_local = int(out_channels * (1 - ratio_global))
        out_global = out_channels - out_local

        self.local_to_local = nn.Conv2d(in_local, out_local, kernel_size=3, padding=1) if in_local > 0 and out_local > 0 else None
        self.local_to_global = nn.Conv2d(in_local, out_global, kernel_size=3, padding=1) if in_local > 0 and out_global > 0 else None
        self.global_to_local = nn.Conv2d(in_global, out_local, kernel_size=3, padding=1) if in_global > 0 and out_local > 0 else None
        self.global_to_global = FourierUnit(in_global, out_global) if in_global > 0 and out_global > 0 else None

        self.out_local = out_local
        self.out_global = out_global

    def forward(self, x):
        if self.ratio_global == 0:
            x_local, x_global = x, torch.zeros_like(x[:, :0])
        else:
            split = int(x.shape[1] * (1 - self.ratio_global))
            x_local, x_global = x[:, :split], x[:, split:]

        out_local, out_global = 0, 0

        if self.local_to_local is not None:
            out_local = out_local + self.local_to_local(x_local)
        if self.global_to_local is not None:
            out_local = out_local + self.global_to_local(x_global)
        if self.local_to_global is not None:
            out_global = out_global + self.local_to_global(x_local)
        if self.global_to_global is not None:
            out_global = out_global + self.global_to_global(x_global)

        if self.out_local > 0 and self.out_global > 0:
            return torch.cat([out_local, out_global], dim=1)
        elif self.out_local > 0:
            return out_local
        else:
            return out_global


class FFCBlock(nn.Module):
    def __init__(self, channels, ratio_global=0.5):
        super(FFCBlock, self).__init__()
        self.ffc1 = FFC(channels, channels, ratio_global)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.ffc2 = FFC(channels, channels, ratio_global)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.ffc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ffc2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out)
        return out


class FFCUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_ffc_blocks=4):
        super(FFCUNet, self).__init__()
        self.encoder = FFCUNetEncoder(in_channels, base_channels)
        self.ffc_blocks = nn.Sequential(*[FFCBlock(base_channels * 8) for _ in range(num_ffc_blocks)])
        self.decoder = FFCUNetDecoder(base_channels, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4, b5 = self.encoder(x)
        b5 = self.ffc_blocks(b5)
        output = self.decoder(b1, b2, b3, b4, b5)
        return output


class FFCUNetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super(FFCUNetEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.MaxPool2d(2)

        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.MaxPool2d(2)

        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        return b1, b2, b3, b4, b5


class FFCUNetDecoder(nn.Module):
    def __init__(self, base_width, out_channels=3):
        super(FFCUNetDecoder, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*16, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))

        self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b1, b2, b3, b4, b5):
        up1 = self.up1(b5)
        db1 = self.db1(torch.cat([up1, b4], dim=1))

        up2 = self.up2(db1)
        db2 = self.db2(torch.cat([up2, b3], dim=1))

        up3 = self.up3(db2)
        db3 = self.db3(torch.cat([up3, b2], dim=1))

        up4 = self.up4(db3)
        db4 = self.db4(torch.cat([up4, b1], dim=1))

        return self.fin_out(db4)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.decoder = UNetDecoder(base_channels, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4, b5, b6)
        return output


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super(UNetEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class UNetDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(UNetDecoder, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out
