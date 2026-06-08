"""Mask convolutional autoencoder model for map mask encoding."""

from typing import Literal

from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from lightning.pytorch.utilities import grad_norm

from rl_detect.model.pl_base import ConfigurableLitModule


class PatchEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11*11, output_size),
        )

    def forward(self, x):
        return self.net(x)


class PatchDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 upsample_arch: Literal['conv_transpose', 'pixel_shuffle']):
        super().__init__()

        if upsample_arch == 'conv_transpose':
            self.net = nn.Sequential(
                nn.Linear(input_size, 11*11),
                nn.Unflatten(1, (1, 11, 11)),
                nn.ConvTranspose2d(1, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0,
                                   output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=0),
                nn.Sigmoid(),
            )

        elif upsample_arch == 'pixel_shuffle':
            self.net = nn.Sequential(
                nn.Linear(input_size, 25*25),
                nn.Unflatten(1, (1, 25, 25)),
                nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.PixelShuffle(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f'Upsample method {upsample_arch} not supported'
                ' (choose from "conv_transpose" or "pixel_shuffle")')

    def forward(self, x):
        return self.net(x)


class MaskConvAutoencoder(nn.Module):
    def __init__(self,
                 bottleneck_size: int,
                 upsample_arch: Literal['conv_transpose', 'pixel_shuffle']):
        super().__init__()

        self.encoder = PatchEncoder(bottleneck_size)
        self.decoder = PatchDecoder(bottleneck_size, upsample_arch)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.GroupNorm(4, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x_BCHW: Tensor):
        skip_BCHW = x_BCHW
        x_BCHW = self.net(x_BCHW)
        x_BCHW = x_BCHW + self.skip_conv(skip_BCHW)
        return x_BCHW


class PatchEncoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            ResidualBlock(16, 16),

            # (B, C, H, W) -> (B, C, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            ResidualBlock(32, 32),

            # (B, C, H/2, W/2) -> (B, C, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            ResidualBlock(64, 64),

            # (B, C, H/4, W/4) -> (B, C, H/8, W/8)
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            ResidualBlock(64, 64),

            nn.GroupNorm(4, 64),
            nn.SiLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.Conv2d(4, 4, kernel_size=1),
        ])

    def forward(self, x_BCHW: Tensor):
        for module in self.module_list:
            if getattr(module, 'stride', None) == (2, 2):
                # Asymmetrical padding (left, right, top, bottom).
                x_BCHW = F.pad(x_BCHW, (0, 1, 0, 1))
            x_BCHW = module(x_BCHW)

        return x_BCHW

class PatchDecoder2(nn.Module):
    def __init__(self, final_size: int):
        super().__init__()

        # Intermediate sizes.
        self.intermediate_sizes = [ final_size // 8,
                                    final_size // 4,
                                    final_size // 2,
                                    final_size ]

        # Upsample with PixelShuffle.

        self.module_list = nn.ModuleList([
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 32 * 4),

            # (B, C, H/8, W/8) -> (B, C, H/4, W/4)
            nn.PixelShuffle(2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 16 * 4),

            # (B, C, H/4, W/4) -> (B, C, H/2, W/2)
            nn.PixelShuffle(2),
            ResidualBlock(16, 16),
            ResidualBlock(16, 8 * 4),

            # (B, C, H/2, W/2) -> (B, C, H, W)
            nn.PixelShuffle(2),
            ResidualBlock(8, 8),
            ResidualBlock(8, 4),

            nn.Conv2d(4, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        ])

    def forward(self, x_BCHW: Tensor):
        upsample_idx = 0
        for module in self.module_list:
            if isinstance(module, nn.PixelShuffle):
                x_BCHW = module(x_BCHW)
                upsample_idx += 1
                curr_spatial_size = x_BCHW.shape[-1]
                if curr_spatial_size != self.intermediate_sizes[upsample_idx]:
                    # Asymmetrical padding (left, right, top, bottom).
                    x_BCHW = F.pad(x_BCHW, (0, 1, 0, 1))

            else:
                x_BCHW = module(x_BCHW)

        return x_BCHW


class MaskConvAutoencoder2(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.encoder = PatchEncoder2()
        self.decoder = PatchDecoder2(input_size)

    def forward(self, x_BCHW: Tensor):
        x_BCHW = self.encoder(x_BCHW)
        x_BCHW = self.decoder(x_BCHW)
        return x_BCHW


class MaskConvAutoencoderLitModule(ConfigurableLitModule):
    def __init__(self,
                 input_size: int,
                 # bottleneck_size: int,
                 # upsample_arch: Literal['conv_transpose', 'pixel_shuffle'],
                 optimizer: dict = None,
                 lr_scheduler: dict = None,
                 early_stopping: dict = None,
                 gradient_clipping: dict = None):

        super().__init__(optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         early_stopping=early_stopping,
                         gradient_clipping=gradient_clipping)

        # self.autoencoder = MaskConvAutoencoder(bottleneck_size=bottleneck_size,
        #                                        upsample_arch=upsample_arch)
        self.autoencoder = MaskConvAutoencoder2(input_size=input_size)
        self.criterion = nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.criterion(x_hat[:, :, :, :], x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.criterion(x_hat[:, :, :, :], x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.criterion(x_hat[:, :, :, :], x)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.autoencoder, norm_type=2)
        self.log_dict(norms)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/loss": 1, "hp/fde": 1})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
