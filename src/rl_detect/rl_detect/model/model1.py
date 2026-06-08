"""Model 1: Trajectory prediction with social context."""
from typing import Literal, Optional

import torch
from torch import nn
from torch import Tensor

from rl_detect.model.modules import (
    TrajectoryEncoderRNN1,
    SocialEncoder1,
    TrajectoryDecoderRNN1,
)
from .social_nce import ISocialNceCompatible
import rl_detect.model.model_utils as model_utils
from rl_detect.model.pl_traj_model import BaseTrajectoryLitModule
from rl_detect.model.sampling_info import SamplingInfo


class MyTrajectoryModel1(nn.Module, ISocialNceCompatible):
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

        self.social_module_hidden_size = social_module_hidden_size

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

        # Decoder hidden size must include noise dimension.
        # So when adapting social module output to decoder input,
        # we need to project to decoder_hidden_size - noise_dim.
        decoder_hidden_size_no_noise = decoder_hidden_size - noise_dim
        self.social_to_decoder_adapter = nn.Linear(social_module_hidden_size,
                                                   decoder_hidden_size_no_noise)

        self.decoder = TrajectoryDecoderRNN1(decoder_arch,
                                             decoder_weights_init,
                                             decoder_hidden_size,
                                             pred_len)

    def forward(self,
                traj_BO2: torch.Tensor,
                scene_idx_B: torch.Tensor,
                num_samples: int = 1,
                noise_type: Literal['local', 'global'] = 'local',
                noise: Optional[torch.Tensor] = None,
                ) -> dict:
        sampling_required = num_samples > 1 or noise is not None
        if sampling_required and self.noise_dim <= 0:
            raise ValueError('Cannot sample multiple trajectories '
                             'without noise')

        batch_size, _, _ = traj_BO2.shape

        # Encode trajectory.
        encoding_BH, _ = self.encoder(traj_BO2)

        # Project encoder output to social module hidden size.
        encoding_adapted_BH = self.encoder_to_social_adapter(encoding_BH)

        # Encode social context.
        social_encoding_BH = self.social_encoder(traj_BO2,
                                                 scene_idx_B,
                                                 encoding_adapted_BH)

        # Project social encoding to decoder hidden size without noise.
        # Shape: (batch_size, decoder_hidden_size - noise_dim)
        social_encoding_adapted_BH = \
            self.social_to_decoder_adapter(social_encoding_BH)

        # Noise handling: sample or use provided noise, making sure
        # it has the correct shape.
        noise_BKL = model_utils.handle_noise(
            batch_size=batch_size,
            num_samples=num_samples,
            scene_idx_B=scene_idx_B,
            noise_dim=self.noise_dim,
            noise_distrib=self.noise_distrib, # type: ignore
            noise_type=noise_type,
            noise=noise,
            device=traj_BO2.device
        )

        # Z = B * K

        # Reshape noise to (batch_size * num_samples, noise_dim),
        # for batch processing.
        noise_ZL = noise_BKL.view(batch_size * num_samples, self.noise_dim)

        # Repeat social encoding for all samples.
        # Shape: (batch_size * num_samples, social_module_hidden_size)
        social_encoding_ZH = social_encoding_adapted_BH.repeat_interleave(num_samples, dim=0)

        # Concatenate noise to social encoding.
        # Shape: (batch_size * num_samples,
        #         social_module_hidden_size + noise_dim)
        decoder_context_ZH = torch.cat((social_encoding_ZH, noise_ZL), dim=1)

        # Repeat last observed position for all samples.
        # Shape: (batch_size * num_samples, 2)
        traj_ZO2 = traj_BO2.repeat_interleave(num_samples, dim=0)

        # Decode trajectory.
        # Shape: (batch_size * num_samples, pred_len, 2)
        output_ZP2 = self.decoder(decoder_context_ZH, traj_ZO2[:, -1, :])

        # Reshape output to (batch_size, num_samples, pred_len, 2)
        output_BKP2 = output_ZP2.view(batch_size, num_samples, self.pred_len, 2)

        return {
            "traj_pred_hat_BKP2": output_BKP2,
            "history_embedding_BH": encoding_BH,
            "social_embedding_BH": social_encoding_BH,
        }

    def social_encoding_size(self) -> int:
        return self.social_module_hidden_size


class Model1LitModule(BaseTrajectoryLitModule):
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

        model = MyTrajectoryModel1(
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
            map_nce_loss_weight=0,
            map_nce_num_contour_points=None,
            map_nce_temperature=None,
            map_nce_proj_size=None,
            env_collision_loss_weight=0,
            goal_net_pretrain_epochs=0,
            goal_loss_weight=0,
            goal_matching_loss_weight=0,
            goal_matching_loss_mode='all',
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            gradient_clipping=gradient_clipping
        )

        self.sampling_info_ = SamplingInfo(
            noise_dim=noise_dim,
            noise_distrib=noise_distrib
        )

        # TODO: remove
        self.encoder_hidden_size = encoder_hidden_size
        self.social_module_hidden_size = social_module_hidden_size

        self.save_hyperparameters()

    def forward(self,
                traj_BO2: torch.Tensor,
                traj_gt_BP2: Optional[torch.Tensor],
                scene_idx_B: torch.Tensor,
                map_mask_B1HW: torch.Tensor = None,
                scene_transform_matrix_B33: torch.Tensor = None,
                homography_2mask_B33: torch.Tensor = None,
                num_samples: int = 1,
                noise_type: Literal['local', 'global'] = 'local',
                noise: Optional[torch.Tensor] = None,
                ) -> dict:

        return self.model(traj_BO2=traj_BO2,
                          scene_idx_B=scene_idx_B,
                          num_samples=num_samples,
                          noise_type=noise_type,
                          noise=noise)

    def loss(self,
             output: dict,
             traj_obs_BO2: Tensor,
             traj_pred_BP2: Tensor) -> Tensor:
        return torch.tensor(0.0, device=traj_obs_BO2.device)

    def sampling_info(self) -> SamplingInfo:
        return self.sampling_info_
