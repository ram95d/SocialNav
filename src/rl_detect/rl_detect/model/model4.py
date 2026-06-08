"""Model 3: like model 1 but with map mask input (and MapNCE)"""

from typing import Literal, Optional

import torch
from torch import nn
import numpy as np
import cv2 as cv

from model.modules import (
    TrajectoryEncoderRNN1,
    SocialEncoder1,
    PositionalEncoding,
    TrajectoryDecoderRNN1,
)
from .social_nce import ISocialNceCompatible
from .map_nce import IMapNceCompatible
import model.model_utils as model_utils
from model.pl_traj_model import BaseTrajectoryLitModule
from model.sampling_info import SamplingInfo


# TODO: remove duplicate code


class MyTrajectoryModel4(nn.Module, ISocialNceCompatible, IMapNceCompatible):
    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 encoder_arch: Literal['gru', 'lstm'],
                 encoder_weights_init: Literal['default', 'custom'],
                 encoder_hidden_size: int,
                 social_module_hidden_size: int,
                 social_info_type: Literal['absolute', 'relative'],
                 social_module_num_layers: int,
                 social_module_nhead: int,
                 social_module_dim_feedforward: int,
                 social_module_dropout: float,
                 social_module_norm_first: bool,
                 social_module_activation: str,
                 environment_encoder_hidden_size: int,
                 decoder_arch: Literal['gru', 'lstm'],
                 decoder_weights_init: Literal['default', 'custom'],
                 decoder_hidden_size: int,
                 noise_dim: int,
                 noise_distrib: Literal['gaussian', 'uniform']):

        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.noise_dim = noise_dim or 0
        self.noise_distrib = noise_distrib or 'gaussian'

        self.encoder_hidden_size = encoder_hidden_size
        self.social_module_hidden_size = social_module_hidden_size

        self.environment_encoder_hidden_size = environment_encoder_hidden_size

        self.encoder = TrajectoryEncoderRNN1(encoder_arch,
                                             encoder_weights_init,
                                             encoder_hidden_size)

        # TODO: maybe parameter to drop adapters if embedding sizes match
        self.encoder_to_social_adapter = nn.Linear(encoder_hidden_size,
                                                   social_module_hidden_size)
        self.encoder_to_env_adapter = nn.Linear(encoder_hidden_size,
                                                environment_encoder_hidden_size)

        self.social_encoder = SocialEncoder1(
            social_info_type=social_info_type,
            num_layers=social_module_num_layers,
            d_model=social_module_hidden_size,
            nhead=social_module_nhead,
            dim_feedforward=social_module_dim_feedforward,
            dropout=social_module_dropout,
            norm_first=social_module_norm_first,
            activation=social_module_activation
        )

        # Load mask encoder.
        # TODO: maybe rename to patch_embedding
        PATCH_SIZE = 10
        PATCH_CHANNELS = 1
        flattened_patch_size = PATCH_SIZE * PATCH_SIZE * PATCH_CHANNELS
        num_patches = 100
        self.sub_patch_size = PATCH_SIZE
        self.patch_encoder = nn.Linear(flattened_patch_size,
                                       encoder_hidden_size)

        # Environment encoder (ViT like).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size))
        self.positional_encoding = PositionalEncoding(encoder_hidden_size,
                                                      dropout=0.1,
                                                      max_len=num_patches + 1)
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=environment_encoder_hidden_size,
            nhead=4,
            dim_feedforward=encoder_hidden_size*4,
            dropout=0.1,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.env_encoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=2
        )

        # Decoder hidden size must include noise dimension
        # and environment encoding.
        # So when adapting social module output to decoder input,
        # we need to project to
        # decoder_hidden_size - noise_dim - environment_encoder_hidden_size
        decoder_hidden_size_pure = decoder_hidden_size \
                                   - noise_dim \
                                   - environment_encoder_hidden_size

        if decoder_hidden_size_pure <= 0:
            raise ValueError('Decoder hidden size is too small '
                             'to accommodate noise and environment encoding')

        self.social_to_decoder_adapter = nn.Linear(social_module_hidden_size,
                                                   decoder_hidden_size_pure)

        self.decoder = TrajectoryDecoderRNN1(decoder_arch,
                                             decoder_weights_init,
                                             decoder_hidden_size,
                                             pred_len)

    def forward(self,
                scene_SO2: torch.Tensor,
                map_mask_1HW: torch.Tensor,
                scene_transform_matrix: torch.Tensor = None,
                homography_2mask: torch.Tensor = None,
                num_samples: int = 1,
                noise_type: Literal['local', 'global'] = 'local',
                noise: Optional[torch.Tensor] = None
                ) -> torch.Tensor \
                   | tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        if num_samples > 1 and self.noise_dim == 0:
            raise ValueError('Cannot sample multiple trajectories '
                             'without noise')

        scene_size, _, _ = scene_SO2.shape

        # Encode trajectory.
        encoding_SH, _ = self.encoder(scene_SO2)

        # Project encoder output to social module hidden size.
        encoding_adapted_SH = self.encoder_to_social_adapter(encoding_SH)

        # Encode social context.
        social_encoding_SH = self.social_encoder(scene_SO2, encoding_adapted_SH)

        # Project social encoding to decoder hidden size without noise.
        # Shape: (scene_size, decoder_hidden_size - noise_dim)
        social_encoding_adapted_SH = \
            self.social_to_decoder_adapter(social_encoding_SH)

        mask_patches_S1HW = model_utils.extract_patches(scene_SO2,
                                                        map_mask_1HW,
                                                        scene_transform_matrix,
                                                        homography_2mask,
                                                        patch_size_px=100,
                                                        back_dist_px=10)

        # F: number of patches
        sub_patches_SF1HW = self._extract_sub_patches(mask_patches_S1HW,
                                                      self.sub_patch_size)

        # Flatten patches.
        # Z = C * H * W
        sub_patches_SFZ = sub_patches_SF1HW.flatten(2)


        # Embed patches.
        patch_embedding_SFH = self.patch_encoder(sub_patches_SFZ)

        # Add cls token.
        # Shape: (scene_size, F + 1, hidden_size)
        patch_embedding_SFH = torch.cat(
            (self.cls_token.expand(scene_size, -1, -1), patch_embedding_SFH),
            dim=1
        )

        # Memory for transformer decoder.
        history_memory_SH = self.encoder_to_env_adapter(encoding_SH)
        history_memory_S1H = history_memory_SH.unsqueeze(1)

        # Positional encoding of patches.
        patch_embedding_SFH = self.positional_encoding(patch_embedding_SFH)

        # Encode environment.
        patch_embedding_SFH = self.env_encoder(patch_embedding_SFH,
                                               history_memory_S1H)

        # Extract environment encoding from cls token.
        environment_encoding_SH = patch_embedding_SFH[:, 0, :]

        # B = K * S

        # Noise handling: sample or use provided noise, making sure
        # it has the correct shape.
        noise_KSL = model_utils.handle_noise(
            num_samples=num_samples,
            scene_size=scene_size,
            noise_dim=self.noise_dim,
            noise_distrib=self.noise_distrib,
            noise_type=noise_type,
            noise=noise,
            device=scene_SO2.device
        )

        # Reshape noise to (num_samples * scene_size, noise_dim),
        # for batch processing.
        noise_BL = noise_KSL.view(num_samples * scene_size, self.noise_dim)

        # Repeat social encoding for all samples.
        # Shape: (num_samples * scene_size, social_module_hidden_size)
        social_encoding_BH = social_encoding_adapted_SH.repeat(num_samples, 1)

        # Repeat environment encoding for all samples.
        # Shape: (num_samples * scene_size, environment_encoder_hidden_size)
        environment_encoding_BH = environment_encoding_SH.repeat(num_samples, 1)

        # Concatenate social, environment encoding and noise.
        # Shape: (num_samples * scene_size,
        #         social_module_hidden_size + noise_dim)
        decoder_context_BH = torch.cat((social_encoding_BH,
                                        environment_encoding_BH,
                                        noise_BL),
                                       dim=1)

        # Repeat last observed position for all samples.
        # Shape: (num_samples * scene_size, 2)
        scene_BO2 = scene_SO2.repeat(num_samples, 1, 1)

        # Decode trajectory.
        # Shape: (num_samples * scene_size, pred_len, 2)
        output_BP2 = self.decoder(decoder_context_BH, scene_BO2[:, -1, :])

        # Reshape output to (num_samples, scene_size, pred_len, 2)
        output_KSP2 = output_BP2.view(num_samples, scene_size, self.pred_len, 2)

        if self.training:
            # Concatenate social and environment encoding for map NCE loss.
            traj_map_encoding_SH = torch.cat((social_encoding_SH,
                                              environment_encoding_SH),
                                             dim=1)
            return (output_KSP2,
                    encoding_SH,
                    social_encoding_SH,
                    traj_map_encoding_SH,
                    mask_patches_S1HW)

        return output_KSP2

    @torch.no_grad()
    def _extract_sub_patches(self,
                             map_mask_S1HW: torch.Tensor,
                             patch_size: int,
                             ) -> torch.Tensor:
        """Extract sub-patches from mask patches.

        Args:
            map_mask_S1HW: Mask patches tensor. Shape: (scene_size, 1, H, W).

        Returns:
            Sub-patches tensor. Shape: (scene_size, F, 1, H, W).
        """

        B, C, H, W = map_mask_S1HW.shape
        x = map_mask_S1HW.reshape(B, C, H // patch_size, patch_size,
                                        W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x_SF1HW = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        # if flatten_channels:
        #     map_mask_S1HW = map_mask_S1HW.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]

        return x_SF1HW

    def social_encoding_size(self) -> int:
        return self.social_module_hidden_size

    def map_encoding_size(self) -> int:
        return self.environment_encoder_hidden_size + self.encoder_hidden_size


class Model4LitModule(BaseTrajectoryLitModule):
    """PyTorch Lightning model for trajectory prediction."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 # TODO probably make optinal noise related parameters
                 noise_dim: int,
                 noise_distrib: Literal['gaussian', 'uniform'],
                 social_nce_loss_weight: float,
                 social_nce_temperature: Optional[float],
                 social_nce_proj_size: Optional[int],
                 map_nce_loss_weight: float,
                 map_nce_num_contour_points: Optional[int],
                 map_nce_temperature: Optional[float],
                 map_nce_proj_size: Optional[int],
                 env_collision_loss_weight: float,
                 encoder_arch: Literal['gru', 'lstm'],
                 encoder_weights_init: Literal['default', 'custom'],
                 encoder_hidden_size: int,
                 social_module_hidden_size: int,
                 social_info_type: Literal['absolute', 'relative'],
                 social_module_num_layers: int,
                 social_module_nhead: int,
                 social_module_dim_feedforward: int,
                 social_module_dropout: float,
                 social_module_norm_first: bool,
                 # TODO: try also with nn.Module
                 social_module_activation: Literal['relu', 'gelu'],
                 environment_encoder_hidden_size: int,
                 decoder_arch: Literal['gru', 'lstm'],
                 decoder_weights_init: Literal['default', 'custom'],
                 decoder_hidden_size: int,
                 optimizer: dict = None,
                 lr_scheduler: dict = None,
                 early_stopping: dict = None,
                 gradient_clipping: dict = None):
        """Builds the model.

        Args:
            obs_len: The length of the observed trajectory.
            pred_len: The length of the predicted trajectory.
            social_nce_loss_weight: The weight of the social NCE loss.
        """

        model = MyTrajectoryModel4(
            obs_len=obs_len,
            pred_len=pred_len,
            encoder_arch=encoder_arch,
            encoder_weights_init=encoder_weights_init,
            encoder_hidden_size=encoder_hidden_size,
            social_module_hidden_size=social_module_hidden_size,
            social_info_type=social_info_type,
            social_module_num_layers=social_module_num_layers,
            social_module_nhead=social_module_nhead,
            social_module_dim_feedforward=social_module_dim_feedforward,
            social_module_dropout=social_module_dropout,
            social_module_norm_first=social_module_norm_first,
            social_module_activation=social_module_activation,
            environment_encoder_hidden_size=environment_encoder_hidden_size,
            decoder_arch=decoder_arch,
            decoder_weights_init=decoder_weights_init,
            decoder_hidden_size=decoder_hidden_size,
            noise_dim=noise_dim,
            noise_distrib=noise_distrib
        )

        super().__init__(
            model=model,
            obs_len=obs_len,
            pred_len=pred_len,
            num_samples=num_samples,
            social_nce_loss_weight=social_nce_loss_weight,
            social_nce_temperature=social_nce_temperature,
            social_nce_proj_size=social_nce_proj_size,
            map_nce_loss_weight=map_nce_loss_weight,
            map_nce_num_contour_points=map_nce_num_contour_points,
            map_nce_temperature=map_nce_temperature,
            map_nce_proj_size=map_nce_proj_size,
            env_collision_loss_weight=env_collision_loss_weight,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            gradient_clipping=gradient_clipping
        )

        self.sampling_info_ = SamplingInfo(
            noise_dim=noise_dim,
            noise_distrib=noise_distrib
        )

        self.save_hyperparameters()

    def forward(self,
                scene_SO2: torch.Tensor,
                map_mask_1HW: torch.Tensor = None,
                scene_transform_matrix: torch.Tensor = None,
                homography_2mask: torch.Tensor = None,
                num_samples: int = 1,
                noise_type: Literal['local', 'global'] = 'local',
                noise: Optional[torch.Tensor] = None
                ) -> torch.Tensor \
                   | tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:

        return self.model(scene_SO2=scene_SO2,
                          map_mask_1HW=map_mask_1HW,
                          scene_transform_matrix=scene_transform_matrix,
                          homography_2mask=homography_2mask,
                          num_samples=num_samples,
                          noise_type=noise_type,
                          noise=noise)

    def sampling_info(self) -> SamplingInfo:
        return self.sampling_info_
