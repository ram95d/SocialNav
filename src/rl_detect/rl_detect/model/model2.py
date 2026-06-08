from typing import Literal, Optional

import torch
from torch import nn

from rl_detect.model.modules import (
    TrajectoryEncoderRNN1,
    SocialEncoder2,
    TrajectoryDecoderRNN1,
)
from .social_nce import ISocialNceCompatible
import rl_detect.model.model_utils as model_utils
from rl_detect.model.pl_traj_model import BaseTrajectoryLitModule
from rl_detect.model.sampling_info import SamplingInfo


class MyTrajectoryModel2(nn.Module, ISocialNceCompatible):
    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 encoder_arch: Literal['gru', 'lstm'],
                 encoder_weights_init: Literal['default', 'custom'],
                 encoder_hidden_size: int,
                 social_module_hidden_size: int,
                 social_module_num_neighbors: int,
                 social_module_spatial_embedding_size: int,
                 social_module_velocity_embedding_size: int,
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

        # TODO: maybe introduce some non-linearities somewhere
        self.social_encoder = SocialEncoder2(
            hidden_size=social_module_hidden_size,
            num_neighbors=social_module_num_neighbors,
            spatial_embedding_size=social_module_spatial_embedding_size,
            velocity_embedding_size=social_module_velocity_embedding_size
        )

        # TODO: maybe parameter to drop adapters if embedding sizes match

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
                scene_SO2: torch.Tensor,
                num_samples: int = 1,
                noise_type: Literal['local', 'global'] = 'local',
                noise: Optional[torch.Tensor] = None
                ) -> torch.Tensor \
                   | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples > 1 and self.noise_dim == 0:
            raise ValueError('Cannot sample multiple trajectories '
                             'without noise')

        scene_size, _, _ = scene_SO2.shape

        # Encode trajectory.
        encoding_SH, _ = self.encoder(scene_SO2)

        # Encode social context.
        social_encoding_SH = self.social_encoder(scene_SO2[:, -1, :],
                                                 scene_SO2[:, -2, :],
                                                 encoding_SH)

        # Project social encoding to decoder hidden size without noise.
        # Shape: (scene_size, decoder_hidden_size - noise_dim)
        social_encoding_adapted_SH = \
            self.social_to_decoder_adapter(social_encoding_SH)

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

        # Concatenate noise to social encoding.
        # Shape: (num_samples * scene_size,
        #         social_module_hidden_size + noise_dim)
        decoder_context_BH = torch.cat((social_encoding_BH, noise_BL), dim=1)

        # Repeat last observed position for all samples.
        # Shape: (num_samples * scene_size, 2)
        scene_BO2 = scene_SO2.repeat(num_samples, 1, 1)

        # Decode trajectory.
        # Shape: (num_samples * scene_size, pred_len, 2)
        output_BP2 = self.decoder(decoder_context_BH, scene_BO2[:, -1, :])

        # Reshape output to (num_samples, scene_size, pred_len, 2)
        output_KSP2 = output_BP2.view(num_samples, scene_size, self.pred_len, 2)

        if self.training:
            # preds, history_encoding, social_encoding
            return output_KSP2, encoding_SH, social_encoding_SH

        return output_KSP2

    def social_encoding_size(self) -> int:
        return self.social_module_hidden_size


class Model2LitModule(BaseTrajectoryLitModule):
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
                 social_module_num_neighbors: int,
                 social_module_spatial_embedding_size: int,
                 social_module_velocity_embedding_size: int,
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

        model = MyTrajectoryModel2(
            obs_len=obs_len,
            pred_len=pred_len,
            encoder_arch=encoder_arch,
            encoder_weights_init=encoder_weights_init,
            encoder_hidden_size=encoder_hidden_size,
            social_module_hidden_size=social_module_hidden_size,
            social_module_num_neighbors=social_module_num_neighbors,
            social_module_spatial_embedding_size=social_module_spatial_embedding_size,
            social_module_velocity_embedding_size=social_module_velocity_embedding_size,
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
                   | tuple[torch.Tensor, torch.Tensor, None, None, None]:

        output = self.model(scene_SO2=scene_SO2,
                            num_samples=num_samples,
                            noise_type=noise_type,
                            noise=noise)

        if self.training:
            # preds, history_encoding, social_encoding, map_encoding, map_patch
            return output + (None, None)
        else:
            return output

    def sampling_info(self) -> SamplingInfo:
        return self.sampling_info_
