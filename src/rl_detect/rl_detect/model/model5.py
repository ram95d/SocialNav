"""Model 3: like model 1 but with map mask input (and MapNCE)"""

from typing import Literal, Optional

import torch
from torch import nn

from model.modules import (
    TrajectoryEncoderRNN1,
    SocialEncoder1,
    TrajectoryDecoderRNN1
)
from .social_nce import ISocialNceCompatible
from .map_nce import IMapNceCompatible
import model.model_utils as model_utils
from model.pl_traj_model import BaseTrajectoryLitModule
from model.sampling_info import SamplingInfo
from model.modules import PositionalEncoding, TrajectoryDecoderTransformer1

from model.mask_autoenc.mask_autoencoder import PatchEncoder2
from model.goal.simple_goal_net import Simple2GoalNet


class MyTrajectoryModel5(nn.Module, ISocialNceCompatible, IMapNceCompatible):

    # TODO: fix parametrization

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
                 mask_encoder_ckpt: str,

                 patch_size_px: int,
                 back_dist_px: int,
                 # mask_encoder_bottleneck_size: int,
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

        # self.mask_encoder_bottleneck_size = mask_encoder_bottleneck_size
        self.environment_encoder_hidden_size = environment_encoder_hidden_size

        # TODO: should be better parameterized
        self.goal_net = Simple2GoalNet(num_samples=20,
                                       input_size=encoder_hidden_size)

        self.encoder = TrajectoryEncoderRNN1(encoder_arch,
                                             encoder_weights_init,
                                             encoder_hidden_size)

        # TODO: maybe parameter to drop adapters if embedding sizes match
        self.encoder_to_social_adapter = nn.Linear(encoder_hidden_size,
                                                   social_module_hidden_size)

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
        self.patch_encoder = PatchEncoder2()
        self.patch_encoder.requires_grad_(False)
        checkpoint = torch.load(mask_encoder_ckpt, map_location='cpu')
        encoder_weights = {k: v for k, v in checkpoint["state_dict"].items()
                           if k.startswith("autoencoder.encoder.")}
        renamed_weights = {
            k.replace("autoencoder.encoder.", ""): v
            for k, v in encoder_weights.items()
        }
        self.patch_encoder.load_state_dict(renamed_weights)

        # Environment encoder.
        self.patch_size_px = patch_size_px
        self.back_dist_px = back_dist_px
        mask_encoder_bottleneck_spatial_size = patch_size_px // 8
        mask_encoder_bottleneck_channels = 4
        mask_encoder_bottleneck_size = (
            mask_encoder_bottleneck_spatial_size
            * mask_encoder_bottleneck_spatial_size
            * mask_encoder_bottleneck_channels
        )


        SUB_PATCH_SIZE = 5
        SUB_PATCH_CHANNELS = 4
        flattened_patch_size = SUB_PATCH_SIZE * SUB_PATCH_SIZE * SUB_PATCH_CHANNELS
        num_sub_patches = (mask_encoder_bottleneck_spatial_size // SUB_PATCH_SIZE) ** 2
        self.sub_patch_size = SUB_PATCH_SIZE
        self.sub_patch_embedding = nn.Linear(flattened_patch_size,
                                             decoder_hidden_size)

        self.sub_patch_positional_encoding = nn.Embedding(num_sub_patches,
                                                          decoder_hidden_size)


        self.positional_encoding = PositionalEncoding(encoder_hidden_size,
                                                      dropout=0.1,
                                                      max_len=self.pred_len + 1)

        self.decoder = TrajectoryDecoderTransformer1(
            hidden_size=decoder_hidden_size,
            num_layers=3,
            nhead=4,
            dim_feedforward=decoder_hidden_size*4,
            dropout=0,
            norm_first=True,
            activation='gelu',
            pred_len=pred_len
        )

        # TODO: would like this env encoder to depend on the input trajectory
        # such that the model can attend to the relevant parts of the map
        # self.env_encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(mask_encoder_bottleneck_size,
        #               environment_encoder_hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(environment_encoder_hidden_size*2,
        #               environment_encoder_hidden_size)
        # )



        # TODO: test simpler version
        # self.env_encoder = nn.Linear(mask_encoder_bottleneck_size,
        #                              environment_encoder_hidden_size)

        # Decoder hidden size must include noise dimension
        # and environment encoding.
        # So when adapting social module output to decoder input,
        # we need to project to
        # decoder_hidden_size - noise_dim - environment_encoder_hidden_size
        # decoder_hidden_size_pure = decoder_hidden_size \
        #                            - noise_dim \
        #                            - environment_encoder_hidden_size
        # decoder_hidden_size_pure = decoder_hidden_size \
        #                            - environment_encoder_hidden_size

        # if decoder_hidden_size_pure <= 0:
        #     raise ValueError('Decoder hidden size is too small '
        #                      'to accommodate noise and environment encoding')

        self.social_to_decoder_adapter = nn.Linear(social_module_hidden_size,
                                                   decoder_hidden_size)

        self.encoder_to_decoder_adapter = nn.Linear(encoder_hidden_size,
                                                    decoder_hidden_size)

        self.goal_embedding = nn.Linear(2, decoder_hidden_size)

        # self.decoder = TrajectoryDecoderRNN1(decoder_arch,
        #                                      decoder_weights_init,
        #                                      decoder_hidden_size,
        #                                      pred_len)

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
                           torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples > 1 and self.noise_dim == 0:
            raise ValueError('Cannot sample multiple trajectories '
                             'without noise')

        scene_size, _, _ = scene_SO2.shape

        # Encode trajectory.
        encoding_SH, _ = self.encoder(scene_SO2)

        goal_delta_SK2 = self.goal_net(encoding_SH)
        goal_delta_KS2 = goal_delta_SK2.permute(1, 0, 2)


        # ################################################################
        # last_pos_S2 = scene_SO2[:, -1, :]
        # goal_KS2 = goal_delta_KS2 + last_pos_S2[None, :, :]
        # K = goal_KS2.shape[0]
        # S = goal_KS2.shape[1]
        # P = self.pred_len
        # if self.training:
        #     return (torch.zeros(K, S, P, 2),
        #             encoding_SH,
        #             torch.zeros(S, self.social_module_hidden_size).to(scene_SO2.device),
        #             torch.zeros(S, self.encoder_hidden_size + self.environment_encoder_hidden_size).to(scene_SO2.device),
        #             torch.zeros(S, 1, 100, 100).to(scene_SO2.device),
        #             goal_KS2)
        # else:
        #     return torch.zeros(K, S, P, 2)

        # ################################################################

        # Project encoder output to social module hidden size.
        # encoding_adapted_SH = self.encoder_to_social_adapter(encoding_SH)

        # Encode social context.
        # social_encoding_SH = self.social_encoder(scene_SO2, encoding_adapted_SH)

        # Project social encoding to decoder hidden size without noise.
        # Shape: (scene_size, decoder_hidden_size - noise_dim)
        # social_encoding_adapted_SH = \
        #     self.social_to_decoder_adapter(social_encoding_SH)

        mask_patches_S1HW = model_utils.extract_patches(scene_SO2,
                                                        map_mask_1HW,
                                                        scene_transform_matrix,
                                                        homography_2mask,
                                                        patch_size_px=self.patch_size_px,
                                                        back_dist_px=self.back_dist_px)

        # Embed patches.
        patch_embedding_SCHW = self.patch_encoder(mask_patches_S1HW)


        # F: number of patches
        sub_patches_SF1HW = self._extract_sub_patches(patch_embedding_SCHW,
                                                      self.sub_patch_size)

        # Flatten patches.
        # Z = C * H * W
        sub_patches_SFZ = sub_patches_SF1HW.flatten(2)

        # Embed sub-patches.
        # Shape: (scene_size, num_sub_patches, decoder_hidden_size)
        sub_patch_emb_SFH = self.sub_patch_embedding(sub_patches_SFZ)

        # B = K * S

        # Noise handling: sample or use provided noise, making sure
        # it has the correct shape.
        # noise_KSL = model_utils.handle_noise(
        #     num_samples=num_samples,
        #     scene_size=scene_size,
        #     noise_dim=self.noise_dim,
        #     noise_distrib=self.noise_distrib,
        #     noise_type=noise_type,
        #     noise=noise,
        #     device=scene_SO2.device
        # )

        # Reshape noise to (num_samples * scene_size, noise_dim),
        # for batch processing.
        # noise_BL = noise_KSL.view(num_samples * scene_size, self.noise_dim)


        encoding_SH = self.encoder_to_decoder_adapter(encoding_SH)

        # Repeat social encoding for all samples.
        # Shape: (num_samples * scene_size, social_module_hidden_size)
        encoding_BH = encoding_SH.repeat(num_samples, 1)

        # Repeat environment encoding for all samples.
        # Shape: (num_samples * scene_size, environment_encoder_hidden_size)
        sub_patch_emb_BFH = sub_patch_emb_SFH.repeat(num_samples, 1, 1)

        # Concatenate social, environment encoding and noise.
        # Shape: (num_samples * scene_size,
        #         social_module_hidden_size
        #         + environment_encoder_hidden_size
        #         + noise_dim)
        # decoder_context_BH = torch.cat((social_encoding_BH,
        #                                 environment_encoding_BH),
        #                                 # noise_BL),
        #                                dim=1)

        # Repeat last observed position for all samples.
        # Shape: (num_samples * scene_size, 2)
        scene_BO2 = scene_SO2.repeat(num_samples, 1, 1)
        goal_delta_B2 = goal_delta_KS2.reshape(num_samples * scene_size, -1)

        # Decode trajectory.
        # Shape: (num_samples * scene_size, pred_len, 2)
        # TODO: is this the right way of repeating the goal?
        last_pos_B2 = scene_BO2[:, -1, :]
        # TODO: detach goal_delta?
        goal_delta_detached_B2 = goal_delta_B2.detach()

        goal_delta_detached_BH = self.goal_embedding(goal_delta_detached_B2)

        decoder_context_BH = encoding_BH + goal_delta_detached_BH

        output_BP2 = self.decoder(decoder_context_BH, sub_patch_emb_BFH, last_pos_B2)

        # Reshape output to (num_samples, scene_size, pred_len, 2)
        output_KSP2 = output_BP2.view(num_samples, scene_size, self.pred_len, 2)
        last_pos_S2 = scene_SO2[:, -1, :]
        goal_KS2 = goal_delta_KS2 + last_pos_S2[None, :, :]

        if self.training:
            # Concatenate history and environment encoding for map NCE loss.
            # TODO: think about what is better to do here
            traj_map_encoding_SH = torch.zeros(scene_size, encoding_SH.shape[1] + self.environment_encoder_hidden_size).to(scene_SO2.device)
            social_encoding_SH = torch.zeros(scene_size, self.social_module_hidden_size).to(scene_SO2.device)

            return (output_KSP2,
                    goal_KS2,
                    encoding_SH,
                    social_encoding_SH,
                    traj_map_encoding_SH,
                    mask_patches_S1HW)

        return output_KSP2

    def social_encoding_size(self) -> int:
        return self.social_module_hidden_size

    def map_encoding_size(self) -> int:
        return self.environment_encoder_hidden_size + self.encoder_hidden_size

    @torch.no_grad()
    def _extract_sub_patches(self,
                             map_mask_SCHW: torch.Tensor,
                             patch_size: int,
                             ) -> torch.Tensor:
        """Extract sub-patches from mask patches.

        Args:
            map_mask_S1HW: Mask patches tensor. Shape: (scene_size, 1, H, W).

        Returns:
            Sub-patches tensor. Shape: (scene_size, F, 1, H, W).
        """

        B, C, H, W = map_mask_SCHW.shape
        x = map_mask_SCHW.reshape(B, C, H // patch_size, patch_size,
                                        W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x_SFCHW = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        # if flatten_channels:
        #     map_mask_S1HW = map_mask_S1HW.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]

        return x_SFCHW


class Model5LitModule(BaseTrajectoryLitModule):
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
                 mask_encoder_ckpt: str,

                 patch_size_px: int,
                 back_dist_px: int,
                 # mask_encoder_bottleneck_size: int,
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

        model = MyTrajectoryModel5(
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
            mask_encoder_ckpt=mask_encoder_ckpt,

            patch_size_px=patch_size_px,
            back_dist_px=back_dist_px,
            # mask_encoder_bottleneck_size=mask_encoder_bottleneck_size,
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
