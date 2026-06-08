"""Goal sampling network."""

import math
from typing import Optional, Literal
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.goal.pl_goal_model import BaseGoalNetLitModule
import rl_detect.utils as utils
import model.model_utils as model_utils
from model.pl_base import ConfigurableLitModule
from model.modules import TrajectoryEncoderRNN1, PositionalEncoding
from model import metrics
from model.goal.simple_goal_net import Simple3GoalNet


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm_groups=16):
        super().__init__()

        self.main = nn.Sequential(
            nn.GroupNorm(group_norm_groups, in_channels) if group_norm_groups > 0 else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_norm_groups, out_channels) if group_norm_groups > 0 else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # If the input and output channels are different,
        # apply a 1x1 convolution to the skip connection.
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x_BCHW):
        skip_BCHW = x_BCHW

        x_BCHW = self.main(x_BCHW)

        return x_BCHW + self.skip_conv(skip_BCHW)

class FullEncoderBlock(nn.Module):
    def __init__(self,
                 downsampler: Literal['conv', 'maxpool'],
                 in_channels: int,
                 out_channels: int,
                 num_residual_blocks: int,
                 group_norm_groups: int = 16):
        super().__init__()

        if downsampler == 'conv':
            self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif downsampler == 'maxpool':
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            raise ValueError(f'Invalid downsampler: {downsampler}')

        self.residual_blocks = nn.ModuleList([ResidualBlock(out_channels, out_channels, group_norm_groups) for _ in range(num_residual_blocks)])

    def forward(self, x_BCHW):
        x_BCHW = self.downsampler(x_BCHW)
        for block in self.residual_blocks:
            x_BCHW = block(x_BCHW)
        return x_BCHW

class FullDecoderBlock(nn.Module):
    def __init__(self,
                 upsampler: Literal['convt', 'upsample'],
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 num_residual_blocks: int,
                 group_norm_groups: int = 16):
        super().__init__()

        self.skip_conv = nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=1, padding=0)

        self.residual_blocks = nn.ModuleList([ResidualBlock(in_channels, in_channels, group_norm_groups) for _ in range(num_residual_blocks)])

        if upsampler == 'convt':
            self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif upsampler == 'upsample':
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f'Invalid upsampler: {upsampler}')

    def forward(self, x_BCHW: Tensor):
        x_BCHW = self.skip_conv(x_BCHW)
        for block in self.residual_blocks:
            x_BCHW = block(x_BCHW)
        x_BCHW = self.upsampler(x_BCHW)
        return x_BCHW

class SampleHeads(nn.Module):
    def __init__(self, num_channels, num_samples):
        super().__init__()

        self.num_samples = num_samples
        self.num_channels = num_channels

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            ) for _ in range(num_samples)
        ])

        self.final = nn.Conv2d(num_channels, 1, kernel_size=1)

    def forward(self, x_BCHW):
        skip_BCHW = x_BCHW
        x_BCHW_list = [ (head(x_BCHW) + skip_BCHW) for head in self.heads ]
        x_B1HW_list = [ self.final(x_BCHW) for x_BCHW in x_BCHW_list ]
        x_BKHW = torch.cat(x_B1HW_list, dim=1)

        return x_BKHW

# TODO: parametrize (number of channels, upsampling method, bottleneck size, use attention, "final" architecture, activations, normalization, etc.)

class MapGoalNet(nn.Module):
    def __init__(self,
                 obs_len: int,
                 num_samples: int,
                 map_size: int,
                 max_map_size: int,
                 kernel_length: int,
                 sigma: float,
                 encoder_channels: list[int],
                 bottleneck_channels: list[int],
                 decoder_channels: list[int],
                 downsampler: Literal['conv', 'maxpool'],
                 upsampler: Literal['convt', 'upsample'],
                 num_residual_blocks: int,
                 encoder_norm_groups: list[int] | None,
                 decoder_norm_groups: list[int] | None):

        super().__init__()

        num_downsampling = len(encoder_channels) - 1
        if map_size % (2**num_downsampling) != 0:
            raise ValueError(f'Map size must be divisible by 2^{num_downsampling}')

        # TODO: better to find the smallest max_map_size

        if encoder_norm_groups is None:
            encoder_norm_groups = [0] * (len(encoder_channels) - 1)
        if decoder_norm_groups is None:
            decoder_norm_groups = [0] * len(decoder_channels)

        if len(encoder_norm_groups) != len(encoder_channels) - 1:
            raise ValueError(f'Expected {len(encoder_channels) - 1} encoder norm groups, got {len(encoder_norm_groups)}')
        if len(decoder_norm_groups) != len(decoder_channels):
            raise ValueError(f'Expected {len(decoder_channels)} decoder norm groups, got {len(decoder_norm_groups)}')

        # Add 0 to the first element of encoder_norm_groups since the first layer has no normalization.
        encoder_norm_groups = [0] + encoder_norm_groups


        self.obs_len = obs_len
        self.num_samples = num_samples
        self.map_size = map_size    # Size of the downsampled map
        self.max_map_size = max_map_size
        self.scale_factor = self.map_size / self.max_map_size
        self.kernel_length = kernel_length
        self.sigma = sigma

        input_channels = obs_len + 1  # obs_len past positions + 1 map mask
        template_size = int(self.map_size * 2.5)

        # Precompute templates to later extract patches from.
        self.register_buffer('position_template_HW',
                             create_distance_heatmap_template(template_size,
                                                              normalize=True))

        self.register_buffer('gt_template_HW',
                             create_gaussian_heatmap_template(template_size,
                                                              kernel_length=kernel_length,
                                                              sigma=sigma,
                                                              normalize=False))

        # Build encoder
        self.encoder = nn.ModuleList()
        curr_channels = input_channels

        # First conv layer (no downsampling)
        self.encoder.append(
            nn.Conv2d(curr_channels, encoder_channels[0], kernel_size=3, padding=1)
        )
        curr_channels = encoder_channels[0]

        # Remaining encoder layers (with downsampling)
        for out_channels, norm_groups in zip(encoder_channels[1:], encoder_norm_groups[1:]):
            self.encoder.append(
                FullEncoderBlock(
                    downsampler=downsampler,
                    in_channels=curr_channels,
                    out_channels=out_channels,
                    num_residual_blocks=num_residual_blocks,
                    group_norm_groups=norm_groups
                )
            )
            curr_channels = out_channels

        self.positional_encoding = PositionalEncoding(32, dropout=0.0, max_len=(self.map_size//8)**2)

        # Build bottleneck
        self.bottleneck = nn.ModuleList()

        # Intermediate layers (conv + activation)
        for out_channels in bottleneck_channels[:-1]:
            self.bottleneck.append(
                nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)
            )
            self.bottleneck.append(nn.SiLU())
            curr_channels = out_channels

        # Final layer (just conv, no activation)
        self.bottleneck.append(
            nn.Conv2d(curr_channels, bottleneck_channels[-1], kernel_size=3, padding=1)
        )
        curr_channels = bottleneck_channels[-1]

        # Build decoder
        self.decoder = nn.ModuleList()
        encoder_features = encoder_channels[::-1]  # Reverse encoder channels for skip connections
        for (out_channels, skip_channels, norm_groups) in zip(decoder_channels, encoder_features, decoder_norm_groups):
            self.decoder.append(
                FullDecoderBlock(
                    upsampler=upsampler,
                    in_channels=curr_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    num_residual_blocks=num_residual_blocks,
                    group_norm_groups=norm_groups
                )
            )
            curr_channels = out_channels

        # Final processing
        self.decoder.append(
            nn.Conv2d(curr_channels + encoder_channels[0], self.num_samples*2, kernel_size=3, padding=1)
        )

        self.final = nn.Sequential(
            ResidualBlock(self.num_samples*2, self.num_samples*2, group_norm_groups=0),
            nn.Conv2d(self.num_samples*2, self.num_samples, kernel_size=1),
        )

    def forward(self,
                traj_BO2: Tensor,
                map_mask_B1HW: Tensor,
                hom_meters2mask_B33: Tensor,
                hom_mask2meters_B33: Tensor):

        # Meters to mask coordinates.
        traj_mask_BO2 = utils.project_batched(traj_BO2, hom_meters2mask_B33)
        traj_mask_BO2 = traj_mask_BO2 * self.scale_factor

        # Prepare the map masks.
        _, _, H, W = map_mask_B1HW.shape
        pad_h = self.max_map_size - H
        pad_w = self.max_map_size - W
        # TODO: understand with what to pad (0, 1, 255)
        map_mask_scaled_B1HW = F.pad(map_mask_B1HW, (0, pad_w, 0, pad_h))
        map_mask_scaled_B1HW = F.interpolate(map_mask_scaled_B1HW,
                                             size=(self.map_size, self.map_size))

        # Create gaussian heatmaps.
        traj_heatmaps_BOHW = self._create_traj_heatmaps(traj_mask_BO2, self.map_size)

        # Concatenate heatmaps with map masks.
        map_mask_BCHW = torch.cat([map_mask_scaled_B1HW, traj_heatmaps_BOHW], dim=1)

        # Forward pass.
        x_BCHW = map_mask_BCHW
        skip_connections = []
        for module in self.encoder:
            x_BCHW = module(x_BCHW)
            skip_connections.append(x_BCHW)

        for module in self.bottleneck:
            if isinstance(module, nn.MultiheadAttention):
                x_BCT = x_BCHW.flatten(start_dim=2)
                x_BTC = x_BCT.permute(0, 2, 1)
                x_BTC = self.positional_encoding(x_BTC)
                x_BTC, _ = module(x_BTC, x_BTC, x_BTC)
                x_BCHW = x_BTC.permute(0, 2, 1).view_as(x_BCHW)
            else:
                x_BCHW = module(x_BCHW)

        for module in self.decoder:
            x_BCHW = torch.cat([x_BCHW, skip_connections.pop()], dim=1)
            x_BCHW = module(x_BCHW)

        x_BCHW = self.final(x_BCHW)
        # x_BCHW = self.sample_heads(x_BCHW)

        # Pixel-wise probabilities.
        heatmap_BKHW = x_BCHW
        # heatmap_BKHW = F.sigmoid(heatmap_BKHW)

        # Extract goals (argmax).
        goal_mask_BK2 = self._extract_goals(heatmap_BKHW)

        # Mask to meters coordinates.
        goal_mask_BK2 = goal_mask_BK2 / self.scale_factor
        goal_BK2 = utils.project_batched(goal_mask_BK2, hom_mask2meters_B33)

        # Compute delta goals.
        # last_pos_B12 = traj_BO2[:, -1].unsqueeze(1)
        # delta_goal_BK2 = goal_BK2 - last_pos_B12

        return goal_BK2, heatmap_BKHW, map_mask_scaled_B1HW

    def _create_traj_heatmaps(self, traj_mask_BO2: Tensor, map_size: int):
        """Create heatmaps from trajectories.

        Args:
            traj_mask_BO2: Trajectories in mask coordinates. Shape: (batch_size, obs_len, 2)
            map_size: Size of the square map (H=W)

        Returns:
            Tensor of shape (batch_size, obs_len, map_size, map_size) containing distance-based heatmaps
        """

        batch_size = traj_mask_BO2.size(0)

        # Merge batch and obs_len dimensions.
        traj_mask_Z2 = traj_mask_BO2.view(-1, 2)

        # Gaussian heatmaps for the trajectories.
        heatmap_BHW = heatmap_patch(self.position_template_HW, traj_mask_Z2, map_size, map_size)

        # Reshape to (B, O, H, W).
        heatmaps_BOHW = heatmap_BHW.view(batch_size, -1, map_size, map_size)

        return heatmaps_BOHW

    def _gt_heatmaps(self, gt_goal_mask_B2: Tensor, map_size: int):
        """Create ground truth heatmaps from goals.

        Args:
            gt_goal_mask_B2: Ground truth goals in mask coordinates. Shape: (batch_size, 2)
            map_size: Size of the square map (H=W)

        Returns:
            Tensor of shape (batch_size, map_size, map_size).
        """

        # Gaussian heatmaps for the goals.
        gt_heatmaps_BHW = heatmap_patch(self.gt_template_HW, gt_goal_mask_B2, map_size, map_size)

        return gt_heatmaps_BHW


    def _extract_goals(self, out_BKHW: Tensor):
        """Extract goals from the output of the network.

        Args:
            out_BKHW: Output of the network. Shape: (batch_size, num_samples, map_size, map_size)

        Returns:
            Tensor of shape (batch_size, num_samples, 2) containing the goals
        """

        # Find the maximum value in the heatmap of each sample.

        # Flatten spatial dimensions (S=H*W).
        out_BKS = out_BKHW.flatten(start_dim=2)
        flat_index_BK = out_BKS.argmax(dim=-1)

        # Compute the row and column indices.
        x_index_BK = flat_index_BK // out_BKHW.size(-1)
        y_index_BK = flat_index_BK % out_BKHW.size(-1)

        # Convert row and column indices to coordinates.
        return torch.stack([y_index_BK, x_index_BK], dim=-1).float()


    # def loss(self, goal_BK2: Tensor, out_BKHW: Tensor, gt_goal_B2: Tensor, teacher_goal_BK2: Tensor, hom_meters2mask: Tensor, map_mask_B1HW: Tensor):
    #     """Compute the loss of the network.

    #     Args:
    #         goals_BK2: Ground truth goals. Shape: (batch_size, num_samples, 2)
    #         out_BKHW: Output of the network. Shape: (batch_size, num_samples, map_size, map_size)

    #     Returns:
    #         Tensor containing the loss.
    #     """

    #     batch_size = goal_BK2.size(0)

    #     # GT goals to mask coordinates.
    #     gt_goal_mask_B2 = utils.project_batched(gt_goal_B2, hom_meters2mask)
    #     gt_goal_mask_B2 = gt_goal_mask_B2 * self.scale_factor

    #     # Compute the ground truth heatmaps.
    #     gt_heatmaps_BHW = self._gt_heatmaps(gt_goal_mask_B2, self.map_size)

    #     # Best goal indices.
    #     best_goal_sample_index_B = \
    #         closest_sample_index(goal_BK2, gt_goal_B2)

    #     # Best goal heatmaps.
    #     best_goal_heatmap_BHW = out_BKHW[torch.arange(batch_size),
    #                                      best_goal_sample_index_B]

    #     # Compute the loss.
    #     if self.training and self.current_epoch < self.pretrain_epochs:
    #         # Teacher goals to mask coordinates.
    #         teacher_goal_mask_BK2 = utils.project_batched(teacher_goal_BK2, hom_meters2mask)
    #         teacher_goal_mask_BK2 = teacher_goal_mask_BK2 * self.scale_factor

    #         # TODO: first try only teacher loss then fix it with
    #         # TODO: first try only teacher loss then fix it with
    #         # TODO: first try only teacher loss then fix it with
    #         # TODO: first try only teacher loss then fix it with

    #         # Teacher goal heatmaps.
    #         teacher_goal_mask_Z2 = teacher_goal_mask_BK2.view(-1, 2)
    #         teacher_goal_heatmap_ZHW = self._gt_heatmaps(teacher_goal_mask_Z2, self.map_size)
    #         teacher_goal_heatmap_BKHW = \
    #             teacher_goal_heatmap_ZHW.view(batch_size, self.num_samples, self.map_size, self.map_size)
    #         # not_obstacle_B1HW = map_mask_B1HW > 0.5
    #         # teacher_goal_heatmap_BKHW = teacher_goal_heatmap_BKHW * not_obstacle_B1HW

    #         # # Substitute the best teacher heatmap with the ground truth heatmap.
    #         # best_teacher_sample_index_B = \
    #         #     closest_sample_index(teacher_goal_BK2, gt_goal_B2)
    #         # teacher_goal_heatmap_BKHW[torch.arange(batch_size), best_goal_sample_index_B] = gt_heatmaps_BHW

    #         # Loss for teacher samples.
    #         loss_teacher = F.l1_loss(out_BKHW, teacher_goal_heatmap_BKHW)

    #         # # Other goals loss.
    #         # gt_heatmaps_BKHW = map_mask_B1HW.expand(-1, self.num_samples, -1, -1)
    #         # is_obstacle_B1HW = map_mask_B1HW < 0.5
    #         # is_obstacle_BKHW = is_obstacle_B1HW.expand(-1, self.num_samples, -1, -1)
    #         # loss_others = F.mse_loss(out_BKHW[is_obstacle_BKHW], gt_heatmaps_BKHW[is_obstacle_BKHW])

    #         # Total loss.
    #         loss = loss_teacher
    #     else:
    #         # Best goal loss.
    #         loss_best = F.mse_loss(best_goal_heatmap_BHW, gt_heatmaps_BHW)

    #         # Other goals loss.
    #         gt_heatmaps_BKHW = map_mask_B1HW.expand(-1, self.num_samples, -1, -1)
    #         is_obstacle_B1HW = map_mask_B1HW < 0.5
    #         is_obstacle_BKHW = is_obstacle_B1HW.expand(-1, self.num_samples, -1, -1)
    #         loss_others = F.mse_loss(out_BKHW[is_obstacle_BKHW], gt_heatmaps_BKHW[is_obstacle_BKHW])

    #         # Total loss.
    #         loss = loss_best + loss_others

    #     return loss, best_goal_sample_index_B

    # def loss_debug(self, goal_BK2: Tensor, out_BKHW: Tensor, gt_goal_B2: Tensor, hom_meters2mask: Tensor, debug_traj_BO2: Tensor, debug_map_mask_B1HW: Tensor):
    #     """Compute the loss of the network.

    #     Args:
    #         goals_BK2: Ground truth goals. Shape: (batch_size, num_samples, 2)
    #         out_BKHW: Output of the network. Shape: (batch_size, num_samples, map_size, map_size)

    #     Returns:
    #         Tensor containing the loss.
    #     """

    #     batch_size = goal_BK2.size(0)

    #     # GT goals to mask coordinates.
    #     gt_goal_mask_B2 = utils.project_batched(gt_goal_B2, hom_meters2mask)
    #     gt_goal_mask_B2 = gt_goal_mask_B2 * self.scale_factor

    #     # Compute the ground truth heatmaps.
    #     gt_heatmap_BHW = self._gt_heatmaps(gt_goal_mask_B2, self.map_size)

    #     # Best goal indices.
    #     best_goal_sample_index_B = \
    #         closest_sample_index(goal_BK2, gt_goal_B2)

    #     # Best goal heatmaps.
    #     best_goal_heatmap_BHW = out_BKHW[torch.arange(batch_size),
    #                                      best_goal_sample_index_B]

    #     # Compute the loss.
    #     loss = F.mse_loss(best_goal_heatmap_BHW, gt_heatmap_BHW)

    #     # Display (gt heatmap, best goal heatmap, last_pos heatmap, map mask)
    #     debug_traj_mask_BO2 = utils.project_batched(debug_traj_BO2, hom_meters2mask)
    #     debug_traj_mask_BO2 = debug_traj_mask_BO2 * self.scale_factor
    #     debug_last_pos_B2 = debug_traj_mask_BO2[:, -1]
    #     debug_last_pos_heatmap_BHW = self._create_traj_heatmaps(debug_last_pos_B2, self.map_size)[:, 0]
    #     best_goal_B2 = goal_BK2[torch.arange(batch_size), best_goal_sample_index_B]
    #     best_goal_mask_B2 = utils.project_batched(best_goal_B2, hom_meters2mask)
    #     best_goal_mask_B2 = best_goal_mask_B2 * self.scale_factor
    #     all_goal_mask_BK2 = utils.project_batched(goal_BK2, hom_meters2mask)
    #     all_goal_mask_BK2 = all_goal_mask_BK2 * self.scale_factor
    #     fig, axs = plt.subplots(2, 2)
    #     axs[0, 0].imshow(gt_heatmap_BHW[0].cpu().detach().numpy())
    #     axs[0, 1].imshow(best_goal_heatmap_BHW[0].cpu().detach().numpy())
    #     # colorbar
    #     cbar = plt.colorbar(axs[0, 1].imshow(best_goal_heatmap_BHW[0].cpu().detach().numpy()), ax=axs[0, 1])
    #     cbar.set_label('Best goal heatmap')
    #     axs[1, 0].imshow(debug_last_pos_heatmap_BHW[0].cpu().detach().numpy())
    #     # colorbar
    #     cbar = plt.colorbar(axs[1, 0].imshow(debug_last_pos_heatmap_BHW[0].cpu().detach().numpy()), ax=axs[1, 0])
    #     cbar.set_label('Last pos heatmap')
    #     axs[1, 1].imshow(debug_map_mask_B1HW[0, 0].cpu().detach().numpy(), cmap='gray')
    #     # last pos blue
    #     axs[1, 1].scatter(debug_last_pos_B2[0, 0].cpu().detach().numpy(), debug_last_pos_B2[0, 1].cpu().detach().numpy(), color='blue')
    #     # gt goal green
    #     axs[1, 1].scatter(gt_goal_mask_B2[0, 0].cpu().detach().numpy(), gt_goal_mask_B2[0, 1].cpu().detach().numpy(), color='green')
    #     # all goals orange
    #     axs[1, 1].scatter(all_goal_mask_BK2[0, :, 0].cpu().detach().numpy(), all_goal_mask_BK2[0, :, 1].cpu().detach().numpy(), color='orange', s=10)
    #     # best goal red
    #     axs[1, 1].scatter(best_goal_mask_B2[0, 0].cpu().detach().numpy(), best_goal_mask_B2[0, 1].cpu().detach().numpy(), color='red')
    #     plt.show()

    #     # fig, axs = plt.subplots(1, 1)
    #     # map_mask_HW = debug_map_mask_B1HW[0, 0].cpu().detach().numpy()
    #     # map_mask_HW = cv.resize(map_mask_HW, (1000, 1000), interpolation=cv.INTER_NEAREST)
    #     # axs.imshow(map_mask_HW, cmap='gray')
    #     # scale = 1000 / self.map_size
    #     # # last pos blue
    #     # axs.scatter(debug_last_pos_B2[0, 0].cpu().detach().numpy() * scale, debug_last_pos_B2[0, 1].cpu().detach().numpy() * scale, color='blue')
    #     # # gt goal green
    #     # axs.scatter(gt_goal_mask_B2[0, 0].cpu().detach().numpy() * scale, gt_goal_mask_B2[0, 1].cpu().detach().numpy() * scale, color='green')
    #     # # all goals orange
    #     # axs.scatter(all_goal_mask_BK2[0, :, 0].cpu().detach().numpy() * scale, all_goal_mask_BK2[0, :, 1].cpu().detach().numpy() * scale, color='orange', s=10)
    #     # # best goal red
    #     # axs.scatter(best_goal_mask_B2[0, 0].cpu().detach().numpy() * scale, best_goal_mask_B2[0, 1].cpu().detach().numpy() * scale, color='red')
    #     # # zoom (500, 500)
    #     # axs.set_xlim(0, 500)
    #     # axs.set_ylim(0, 500)
    #     # plt.show()



    #     return loss, best_goal_sample_index_B


def gaussian_kernel(length: int, sigma: float, device: str = 'cpu'):
    """Creates a 2D Gaussian kernel of size (length, length) with the given sigma.

    Returns a normalized kernel with values between 0 and 1."""

    ax_L = torch.linspace(-(length - 1) / 2., (length - 1) / 2., length, device=device)
    xx_LL, yy_LL = torch.meshgrid(ax_L, ax_L, indexing='xy')
    kernel_LL = torch.exp(-0.5 * (xx_LL**2 + yy_LL**2) / sigma**2)

    # Normalize between 0 and 1
    kernel_LL = kernel_LL / kernel_LL.max()

    return kernel_LL


def create_gaussian_heatmap_template(size: int,
                                     kernel_length: int,
                                     sigma: float,
                                     normalize: bool,
                                     device: str = 'cpu'):
    """Create a big gaussian heatmap template to later get patches out."""

    template_HW = torch.zeros([size, size], device=device)
    kernel_HW = gaussian_kernel(kernel_length, sigma, device=device)
    m = kernel_HW.shape[0]
    H, W = template_HW.shape
    # Put the gaussian kernel at the center of the template.
    x_low = W // 2 - int(math.floor(m / 2))
    x_up = W // 2 + int(math.ceil(m / 2))
    y_low = H // 2 - int(math.floor(m / 2))
    y_up = H // 2 + int(math.ceil(m / 2))
    template_HW[y_low:y_up, x_low:x_up] = kernel_HW
    if normalize:
        template_HW = template_HW / template_HW.max()
    return template_HW

def create_distance_heatmap_template(size: int, normalize: bool, device: str = 'cpu'):
    """Create a big distance matrix template to later get patches out."""
    middle = size // 2
    # Create coordinate grid.
    ax = torch.arange(size, device=device)
    grid_x_HW, grid_y_HW = torch.meshgrid(ax, ax, indexing='xy')
    grid_coords_HW2 = torch.stack([grid_x_HW, grid_y_HW], dim=-1).float()
    middle_112 = torch.tensor([middle, middle], device=device)[None, None]
    # Compute distances.
    dist_HW = torch.norm(grid_coords_HW2 - middle_112, dim=-1)
    if normalize:
        dist_HW = dist_HW / dist_HW.max() # TODO: * 2 ???
    return dist_HW


def heatmap_patch(template_HW: Tensor, traj_mask_B2: Tensor, patch_height: int, patch_width: int):
    """Extract patches from a heatmap template centered around the trajectory points.

    Args:
        template_HW: The heatmap template. Shape: (H, W)
        traj_mask_B2: The trajectory points. Shape: (batch_size, 2)
        patch_height: The height of the patch.
        patch_width: The width of the patch.

    Returns:
        Tensor of shape (batch_size, patch_height, patch_width) containing patches from the template
    """

    # Round coordinates to integers.
    x_B = torch.round(traj_mask_B2[:, 0]).int()
    y_B = torch.round(traj_mask_B2[:, 1]).int()

    # Calculate patch lower boundaries.
    x_low_B = template_HW.shape[1] // 2 - x_B
    y_low_B = template_HW.shape[0] // 2 - y_B

    # Create "base" coordinate grids for extracting patches.
    y_coords_H = torch.arange(patch_height, device=template_HW.device)
    x_coords_W = torch.arange(patch_width, device=template_HW.device)
    grid_y_BHW, grid_x_BHW = torch.meshgrid(y_coords_H, x_coords_W, indexing='ij')

    # Offset grids for each sample in batch.
    y_offset_B11 = y_low_B[:, None, None]
    x_offset_B11 = x_low_B[:, None, None]
    y_indices_BHW = grid_y_BHW[None, :, :] + y_offset_B11
    x_indices_BHW = grid_x_BHW[None, :, :] + x_offset_B11

    # Extract patches using advanced indexing.
    patches_BHW = template_HW[y_indices_BHW, x_indices_BHW]

    return patches_BHW

################################################################################
################################################################################
################################################################################
################################################################################


# TODO: refactor y-net utility functions (not class methods likely)
# TODO: refactor y-net utility functions (not class methods likely)
# TODO: refactor y-net utility functions (not class methods likely)
# TODO: refactor y-net utility functions (not class methods likely)
# TODO: refactor y-net utility functions (not class methods likely)
# TODO: refactor y-net utility functions (not class methods likely)

# TODO: predict_step (maybe personalized for ynet, or maybe not could be sufficient to save predictions)
# TODO: predict_step (maybe personalized for ynet, or maybe not could be sufficient to save predictions)
# TODO: predict_step (maybe personalized for ynet, or maybe not could be sufficient to save predictions)
# TODO: predict_step (maybe personalized for ynet, or maybe not could be sufficient to save predictions)
# TODO: predict_step (maybe personalized for ynet, or maybe not could be sufficient to save predictions)


class MapGoalNetLitModule(BaseGoalNetLitModule):
    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 map_size: int,
                 max_map_size: int,
                 kernel_length: int,
                 sigma: float,
                 encoder_channels: list[int],
                 bottleneck_channels: list[int],
                 decoder_channels: list[int],
                 downsampler: Literal['conv', 'maxpool'],
                 upsampler: Literal['convt', 'upsample'],
                 num_residual_blocks: int,
                 encoder_norm_groups: Optional[list[int]],
                 decoder_norm_groups: Optional[list[int]],
                 teacher_ckpt: Optional[str],
                 teacher_pretrain_epochs: Optional[int],
                 optimizer: Optional[dict] = None,
                 lr_scheduler: Optional[dict] = None,
                 early_stopping: Optional[dict] = None,
                 gradient_clipping: Optional[dict] = None):


        model = MapGoalNet(
            obs_len=obs_len,
            num_samples=num_samples,
            map_size=map_size,
            max_map_size=max_map_size,
            kernel_length=kernel_length,
            sigma=sigma,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            downsampler=downsampler,
            upsampler=upsampler,
            num_residual_blocks=num_residual_blocks,
            encoder_norm_groups=encoder_norm_groups,
            decoder_norm_groups=decoder_norm_groups,
        )

        super().__init__(
            model=model,
            obs_len=obs_len,
            pred_len=pred_len,
            num_samples=num_samples,

            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            gradient_clipping=gradient_clipping,
        )

        self.teacher_pretrain_epochs = 0
        if teacher_ckpt is not None and teacher_pretrain_epochs is not None:
            # Load teacher model.
            # TODO: avoid hardcoding the model
            self.teacher_goal_net = Simple3GoalNet(obs_len=obs_len,
                                                   num_samples=num_samples,
                                                   layer_sizes=[56, 40])
            checkpoint = torch.load(teacher_ckpt, map_location='cpu')
            model_weights = {k: v for k, v in checkpoint["state_dict"].items()
                               if k.startswith("model.")}
            renamed_weights = {
                k.replace("model.", ""): v
                for k, v in model_weights.items()
            }
            self.teacher_goal_net.load_state_dict(renamed_weights)
            self.teacher_goal_net.requires_grad_(False)
            self.teacher_goal_net.eval()

            self.teacher_pretrain_epochs = teacher_pretrain_epochs

        self.save_hyperparameters()

    def forward(self,
                traj_BO2: Tensor,
                map_mask_B1HW: Tensor,
                hom_meters2mask_B33: Tensor,
                hom_mask2meters_B33: Tensor
                ) -> dict:

        # TODO: make the model return the dict directly
        goal_BK2, heatmap_BKHW, map_mask_scaled_B1HW = \
            self.model(traj_BO2,
                       map_mask_B1HW,
                       hom_meters2mask_B33,
                       hom_mask2meters_B33)

        return {
            "goal_BK2": goal_BK2,
            "heatmap_BKHW": heatmap_BKHW,
            "map_mask_scaled_B1HW": map_mask_scaled_B1HW
        }

    def loss(self,
             output: dict,
             gt_goal_B2: Tensor,
             traj_obs_BO2: Tensor,
             hom_meters2mask_B33) -> Tensor:
        goal_BK2 = output["goal_BK2"]
        heatmap_BKHW = output["heatmap_BKHW"]
        map_mask_scaled_B1HW = output["map_mask_scaled_B1HW"]

        batch_size = goal_BK2.size(0)
        map_size = self.model.map_size
        scale_factor = self.model.scale_factor

        # GT goals to mask coordinates.
        gt_goal_mask_B2 = utils.project_batched(gt_goal_B2, hom_meters2mask_B33)
        gt_goal_mask_B2 = gt_goal_mask_B2 * scale_factor

        # Compute the ground truth heatmaps.
        gt_heatmaps_BHW = self.model._gt_heatmaps(gt_goal_mask_B2, map_size)

        # Best goal indices.
        best_goal_sample_index_B = \
            model_utils.closest_sample_index(goal_BK2, gt_goal_B2, metric='goal')

        # Best goal heatmaps.
        best_goal_heatmap_BHW = heatmap_BKHW[torch.arange(batch_size),
                                             best_goal_sample_index_B]

        # Compute the loss.
        if self.training and self.current_epoch < self.teacher_pretrain_epochs:
            # Compute teacher goals.
            with torch.no_grad():
                teacher_goal_BK2 = self.teacher_goal_net(traj_obs_BO2)

                # Teacher goals to mask coordinates.
                teacher_goal_mask_BK2 = utils.project_batched(teacher_goal_BK2, hom_meters2mask_B33)
                teacher_goal_mask_BK2 = teacher_goal_mask_BK2 * scale_factor

                # TODO: first try only teacher loss then fix it with
                # TODO: first try only teacher loss then fix it with
                # TODO: first try only teacher loss then fix it with
                # TODO: first try only teacher loss then fix it with

                # Teacher goal heatmaps.
                teacher_goal_mask_Z2 = teacher_goal_mask_BK2.view(-1, 2)
                teacher_goal_heatmap_ZHW = self.model._gt_heatmaps(teacher_goal_mask_Z2, map_size)
                teacher_goal_heatmap_BKHW = \
                    teacher_goal_heatmap_ZHW.view(batch_size, self.num_samples, map_size, map_size)
            # not_obstacle_B1HW = map_mask_B1HW > 0.5
            # teacher_goal_heatmap_BKHW = teacher_goal_heatmap_BKHW * not_obstacle_B1HW

            # # Substitute the best teacher heatmap with the ground truth heatmap.
            # best_teacher_sample_index_B = \
            #     closest_sample_index(teacher_goal_BK2, gt_goal_B2)
            # teacher_goal_heatmap_BKHW[torch.arange(batch_size), best_goal_sample_index_B] = gt_heatmaps_BHW

            # Loss for teacher samples.
            loss_teacher = F.l1_loss(heatmap_BKHW, teacher_goal_heatmap_BKHW)

            # # Other goals loss.
            # gt_heatmaps_BKHW = map_mask_B1HW.expand(-1, self.num_samples, -1, -1)
            # is_obstacle_B1HW = map_mask_B1HW < 0.5
            # is_obstacle_BKHW = is_obstacle_B1HW.expand(-1, self.num_samples, -1, -1)
            # loss_others = F.mse_loss(out_BKHW[is_obstacle_BKHW], gt_heatmaps_BKHW[is_obstacle_BKHW])

            # Total loss.
            loss = loss_teacher
        else:
            # Best goal loss.
            loss_best = F.l1_loss(best_goal_heatmap_BHW, gt_heatmaps_BHW)

            # Other goals loss.
            gt_heatmaps_BKHW = map_mask_scaled_B1HW.expand(-1, self.num_samples, -1, -1)
            is_obstacle_B1HW = map_mask_scaled_B1HW < 0.5
            is_obstacle_BKHW = is_obstacle_B1HW.expand(-1, self.num_samples, -1, -1)
            loss_others = F.l1_loss(heatmap_BKHW[is_obstacle_BKHW], gt_heatmaps_BKHW[is_obstacle_BKHW])

            # Total loss.
            loss = loss_best + loss_others

        return loss

################################################################################
################################################################################
################################################################################
################################################################################
















# class _BaseGoalNetLitModule(ConfigurableLitModule):
#     def __init__(self,
#                  obs_len: int,
#                  pred_len: int,
#                  num_samples: int,
#                  input_size: int,
#                  optimizer: Optional[dict] = None,
#                  lr_scheduler: Optional[dict] = None,
#                  early_stopping: Optional[dict] = None,
#                  gradient_clipping: Optional[dict] = None):
#         super().__init__(optimizer, lr_scheduler, early_stopping, gradient_clipping)

#         self.obs_len = obs_len
#         self.pred_len = pred_len

#         # Version 1: goal net conditioned on past positions embeddings.
#         # encoder_arch = 'lstm'
#         # encoder_weights_init = 'default'
#         # encoder_hidden_size = input_size
#         # self.past_encoder = TrajectoryEncoderRNN1(encoder_arch,
#         #                                           encoder_weights_init,
#         #                                           encoder_hidden_size)
#         # self.model = Simple2GoalNet(num_samples, input_size)

#         # Version 2: goal net conditioned on past positions.
#         self.pretrain_epochs = 5
#         self.model = MapGoalNet(num_samples, obs_len)

#         # Load teacher model.
#         # Load mask encoder.
#         checkpoint_path = 'checkpoints/teacher_goal_net.ckpt'
#         self.teacher_goal_net = Simple3GoalNet(num_samples, obs_len)
#         self.teacher_goal_net.requires_grad_(False)
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
#         model_weights = {k: v for k, v in checkpoint["state_dict"].items()
#                            if k.startswith("model.")}
#         renamed_weights = {
#             k.replace("model.", ""): v
#             for k, v in model_weights.items()
#         }
#         self.teacher_goal_net.load_state_dict(renamed_weights)


#     def forward(self,
#                 traj_BO2: Tensor,
#                 map_mask_B1HW: Tensor,
#                 hom_meters2mask_B33: Tensor,
#                 hom_mask2meters_B33: Tensor):

#         goal_BK2, heatmap_BKHW = self.model(traj_BO2, map_mask_B1HW, hom_meters2mask_B33, hom_mask2meters_B33)
#         return goal_BK2, heatmap_BKHW

#     def training_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'train')

#     def validation_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'val')

#     def test_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'test')

#     def _shared_step(self, batch, batch_idx, phase):
#         # Extract data from batch.
#         (traj_orig_bSA2,
#          map_mask_b1HW,
#          transform_matrix_b33,
#          dataset_name_list,
#          homographies_list,
#          coord_system_list) = batch

#         # Remove batch padding, and flatten the scenes in the batch,
#         # so that all trajectories are in the B dimension.
#         # (b, S, A, 2) -> (B=b*S, A, 2)
#         pad_mask_bS = ~(torch.isnan(traj_orig_bSA2)).all(dim=(2, 3))
#         traj_orig_BA2 = traj_orig_bSA2[pad_mask_bS]
#         num_traj = traj_orig_BA2.size(0)

#         # Keep track of the scene each trajectory belongs to.
#         scene_sizes_b = pad_mask_bS.sum(dim=1)
#         scene_idx_B = torch.repeat_interleave(scene_sizes_b)

#         # Fuse the homographies of the scenes into a single tensor.
#         # homographies_list contains a list of dictionaries, where the i-th
#         # dictionary contains the homographies for the i-th scene.
#         hom_orig2meters_B33 = torch.stack([ homographies_list[i]["orig2meters"]
#                                             for i in scene_idx_B ])
#         hom_meters2orig_B33 = torch.stack([ homographies_list[i]["meters2orig"]
#                                             for i in scene_idx_B ])
#         hom_meters2mask_B33 = torch.stack([ homographies_list[i]["meters2mask"]
#                                             for i in scene_idx_B ])
#         hom_mask2meters_B33 = torch.stack([ homographies_list[i]["mask2meters"]
#                                             for i in scene_idx_B ])

#         # Project the scene to meters.
#         traj_BA2 = utils.project_batched(traj_orig_BA2, hom_orig2meters_B33)

#         # Split the scene into observed and predicted trajectories.
#         # traj_BA2 is always of length obs_len + pred_len.
#         traj_obs_BO2 = traj_BA2[:, :self.obs_len]
#         traj_pred_BP2 = traj_BA2[:, -self.pred_len:]
#         # Keep a copy of the ground truth in the original coordinates.
#         traj_pred_orig_BP2 = traj_orig_BA2[:, -self.pred_len:]

#         # Prepare the map mask for the scenes in the batch.
#         map_mask_B1HW = map_mask_b1HW[scene_idx_B]
#         # Prepare the transformation matrix for the scenes in the batch.
#         transform_matrix_B33 = transform_matrix_b33[scene_idx_B]

#         # Compute goals.

#         # Version 1.
#         # goal_delta_BK2 = self(traj_obs_BO2, map_mask_B1HW, hom_meters2mask_B33, hom_meters2orig_B33)
#         # Version 2.
#         # Pad map mask to 1000x1000
#         _, _, H, W = map_mask_B1HW.shape
#         pad_h = 1000 - H
#         pad_w = 1000 - W
#         # TODO: understand with what to pad (0, 1, 255)
#         map_mask_padded_B1HW = F.pad(map_mask_B1HW, (0, pad_w, 0, pad_h))
#         map_mask_padded_B1HW = F.interpolate(map_mask_padded_B1HW, size=(128,128))

#         # map mask from [0, 255] to [0, 1]
#         map_mask_padded_B1HW = map_mask_padded_B1HW / 255.0

#         goal_delta_BK2, heatmap_BKHW = self.forward(traj_obs_BO2, map_mask_padded_B1HW, hom_meters2mask_B33, hom_mask2meters_B33)
#         goal_BK2 = goal_delta_BK2 + traj_obs_BO2[:, -1].unsqueeze(1)
#         goal_orig_BK2 = utils.project_batched(goal_BK2, hom_meters2orig_B33)

#         # Compute teacher goals.
#         with torch.no_grad():
#             # Teacher relative goals in meters.
#             traj_obs_rel_BO2 = traj_obs_BO2.diff(1, dim=1)
#             teacher_goal_BK2, _ = self.teacher_goal_net(traj_obs_rel_BO2)
#             # Teacher absolute goals in meters.
#             teacher_goal_BK2 = teacher_goal_BK2 + traj_obs_BO2[:, -1:]

#         # Compute loss.

#         # TODO: think about delta or not
#         # TODO: think about if passing correct positions (meters or mask)
#         # gt_goal_fake = torch.full_like(traj_pred_BP2[:, -1], 50.0, device=traj_pred_BP2.device)
#         loss, best_goal_sample_index_B = self.model.loss(goal_BK2, heatmap_BKHW, traj_pred_BP2[:, -1], teacher_goal_BK2, hom_meters2mask_B33, map_mask_padded_B1HW)
#         if batch_idx % 100 == 0:
#             _, _ = self.model.loss_debug(goal_BK2, heatmap_BKHW, traj_pred_BP2[:, -1], hom_meters2mask_B33, traj_obs_BO2, map_mask_padded_B1HW)

#         # Compute FDE (in meters).
#         best_goal_B2 = goal_BK2[torch.arange(num_traj), best_goal_sample_index_B]
#         fde_meters = metrics.compute_fde(best_goal_B2[:, None], traj_pred_BP2[:, None, -1]).mean()
#         # Compute FDE (in original coordinates).
#         best_goal_orig_B2 = goal_orig_BK2[torch.arange(num_traj), best_goal_sample_index_B]
#         fde_orig = metrics.compute_fde(best_goal_orig_B2[:, None], traj_pred_orig_BP2[:, None, -1]).mean()

#         # print best goal (in both meters and original coordinates) and ground truth goal for first 5 trajectories
#         for i in range(5):
#             print(f"Trajectory {i}")
#             print(f"Best goal (meters): {best_goal_B2[i].cpu().numpy()}")
#             print(f"Best goal (original): {best_goal_orig_B2[i].cpu().numpy()}")
#             print(f"Ground truth goal (original): {traj_pred_orig_BP2[i, -1].cpu().numpy()}")

#         # Log metrics.
#         log_dict = {
#             f'{phase}_loss': loss,
#             f'{phase}_fde_meters': fde_meters,
#             f'{phase}_fde_orig': fde_orig,
#         }

#         self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_traj)

#         return loss


#     def _shared_step_old(self, batch, batch_idx, phase):
#         num_traj = 0
#         mse_loss = torch.tensor(0.0, device=self.device)
#         for batch_i in zip(*batch):
#             (scene_orig_sA2,
#              map_mask_1HW,
#              transform_matrix,
#              dataset_name,
#              hom_orig2meters,
#              hom_meters2orig,
#              hom_meters2mask,
#              coord_system) = batch_i

#             # Remove batch padding.
#             mask = ~(torch.isnan(scene_orig_sA2)).all(dim=(1, 2))
#             scene_orig_SA2 = scene_orig_sA2[mask]

#             # Project the scene to meters.
#             scene_SA2 = utils.project(scene_orig_SA2, hom_orig2meters)

#             # Split the scene into observed and predicted trajectories.
#             # scene_SA2 is always of length obs_len + pred_len.
#             scene_obs_SO2 = scene_SA2[:, :self.obs_len]
#             scene_pred_SP2 = scene_SA2[:, -self.pred_len:]
#             # Keep a copy of the ground truth in the original coordinates.
#             scene_obs_orig_SO2 = scene_orig_SA2[:, :self.obs_len]
#             scene_pred_orig_SP2 = scene_orig_SA2[:, -self.pred_len:]

#             # Compute goals.
#             goals_delta_BK2 = self(scene_obs_SO2)
#             if self.current_epoch >= 20:
#                 print(goals_delta_BK2.norm(dim=-1).max().item())

#             goals_BK2 = goals_delta_BK2 + scene_obs_SO2[:, -1].unsqueeze(1)

#             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             # if self.current_epoch >= 10:
#             #     # TODO: debug remove
#             #     map_rgb = cv.cvtColor(map_mask_1HW[0].cpu().numpy(), cv.COLOR_GRAY2RGB)
#             #     # project the observed, future and goals to the mask
#             #     scene_obs_mask_SO2 = utils.project(scene_obs_SO2, hom_meters2mask)
#             #     scene_pred_mask_SP2 = utils.project(scene_pred_SP2, hom_meters2mask)
#             #     goals_mask_BK2 = utils.project(goals_BK2, hom_meters2mask)
#             #     # # draw last observed point
#             #     # for i in range(scene_obs_mask_SO2.size(0)):
#             #     #     cv.circle(map_rgb, tuple(scene_obs_mask_SO2[i, -1].int().tolist()), 1, (0, 0, 255), -1)
#             #     # draw observed points for first pedestrian
#             #     for i in range(scene_obs_mask_SO2.size(1)):
#             #         cv.circle(map_rgb, tuple(scene_obs_mask_SO2[0, i].int().tolist()), 1, (0, 0, 255), -1)
#             #     # draw future points for first pedestrian
#             #     for i in range(scene_pred_mask_SP2.size(1)):
#             #         cv.circle(map_rgb, tuple(scene_pred_mask_SP2[0, i].int().tolist()), 1, (255, 0, 0), -1)
#             #     # draw goals for first pedestrian
#             #     for i in range(goals_mask_BK2.size(1)):
#             #         cv.circle(map_rgb, tuple(goals_mask_BK2[0, i].int().tolist()), 1, (0, 255, 0), -1)

#             #     cv.imshow("observed", map_rgb)
#             #     cv.waitKey(0)
#             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#             # Compute loss.

#             # Targets.
#             gt_goals_B2 = scene_pred_SP2[:, -1]

#             best_goal_sample_index_B = \
#             model_utils.closest_sample_index(goals_BK2,
#                                              gt_goals_B2,
#                                              metric='goal')

#             # Extract the best sample for each pedestrian.
#             best_goals_B2 = goals_BK2[torch.arange(goals_BK2.size(0)),
#                                       best_goal_sample_index_B]

#             # Compute loss.
#             # curr_loss = F.mse_loss(best_goals_B2, gt_goals_B2)
#             curr_loss = (best_goals_B2 - gt_goals_B2).norm(dim=-1).sum()
#             mse_loss += curr_loss

#             num_traj += scene_obs_SO2.size(0)

#         # Average loss.
#         loss = mse_loss / num_traj

#         # Log metrics.
#         self.log(f'{phase}_loss',
#                  loss,
#                  prog_bar=True,
#                  on_step=False,
#                  on_epoch=True,
#                  batch_size=num_traj)

#         return loss

