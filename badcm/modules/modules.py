import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (..., H, W) -> (..., H/2, W/2)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    """
    Generator: Unet architecture.
    [Paper]: https://arxiv.org/abs/1505.04597
    [Code Reference]: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),  # (H+1, W+1)
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),   # (H+1, W+1) -> (H, W)
            nn.Tanh(),
        )

        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x, mask=None):
        # U-Net generator with skip connections from encoder to decoder

        x = (x - self.mean) / self.std
        if self.in_channels != 3:
            x = torch.cat([x, mask], dim=1)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)


class GeneratorLarge(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),  # (H+1, W+1)
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),   # (H+1, W+1) -> (H, W)
            nn.Tanh(),
        )

        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x, mask=None):
        # U-Net generator with skip connections from encoder to decoder

        x = (x - self.mean) / self.std
        if mask is not None:
            x = torch.cat([x, mask], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


class Discriminator(nn.Module):
    """
    Discriminator: PatchGAN
    [Paper]: https://arxiv.org/abs/1611.07004
    [Code Reference]: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py
    """
    def __init__(self, in_channels=3, image_size=256):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

        self.patch = (1, image_size // 2 ** 4, image_size // 2 ** 4)

        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.model(x)
        return out
