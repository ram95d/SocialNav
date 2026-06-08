"""Vanilla LSTM model for trajectory prediction.

This model does not consider interactions between agents.
"""

from typing import Literal, Optional

import torch
from torch import nn

from rl_detect.model.modules import TrajectoryEncoderRNN1, TrajectoryDecoderRNN0
from rl_detect.model.pl_traj_model import BaseTrajectoryLitModule
from rl_detect.model.sampling_info import SamplingInfo


class TrajectoryRNN(nn.Module):
    """LSTM model for trajectory prediction."""

    def __init__(self,
                 hidden_size: int,
                 encoder_arch: Literal['gru', 'lstm'],
                 encoder_weights_init: Literal['default', 'custom'],
                 decoder_arch: Literal['gru', 'lstm'],
                 decoder_weights_init: Literal['default', 'custom'],
                 pred_len: int):
        """Builds the model.

        Args:
            hidden_size: Hidden size of the LSTM.
            pred_len: Length of the predicted trajectory.
        """

        super().__init__()

        self.hidden_size = hidden_size

        self.encoder_arch = encoder_arch
        self.encoder = TrajectoryEncoderRNN1(encoder_arch,
                                             encoder_weights_init,
                                             hidden_size)

        self.decoder_arch = decoder_arch
        self.decoder = TrajectoryDecoderRNN0(decoder_arch,
                                             decoder_weights_init,
                                             hidden_size,
                                             pred_len)

    def forward(self,
                scene_SO2: torch.Tensor
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            scene_SO2: Scene tensor. Expected in absolute coordinates.
                Need at least 2 timesteps. Shape: (scene_size, obs_len, 2).

        Returns:
            Predicted scene tensor. Shape: (scene_size, pred_len, 2).
            During training returns a tuple of 2 elements:
                - Predicted scene tensor (as above)
                - History encoding (shape: (scene_size, hidden_size)).
        """

        # Encode trajectory.
        if self.encoder_arch == 'gru':
            hidden_SH, _ = self.encoder(scene_SO2)
            cell_SH = torch.zeros_like(hidden_SH)
        elif self.encoder_arch == 'lstm':
            hidden_SH, cell_SH = self.encoder(scene_SO2)

        # Last delta.
        last_delta_S2 = scene_SO2[:, -1] - scene_SO2[:, -2]

        # Last position.
        last_pos_S2 = scene_SO2[:, -1]

        # Decode trajectory.
        output_SP2 = self.decoder(last_delta_S2, hidden_SH, cell_SH, last_pos_S2)

        # Add sample dimension (always 1 since sampling is not supported).
        output_KSP2 = output_SP2.unsqueeze(0)

        if self.training:
            # preds, history_encoding
            return output_KSP2, hidden_SH

        return output_KSP2


class RNNLitModule(BaseTrajectoryLitModule):
    """PyTorch Lightning model for trajectory prediction."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 hidden_size: int,
                 encoder_arch: Literal['gru', 'lstm'],
                 encoder_weights_init: Literal['default', 'custom'],
                 decoder_arch: Literal['gru', 'lstm'],
                 decoder_weights_init: Literal['default', 'custom'],
                 optimizer: dict = None,
                 lr_scheduler: dict = None,
                 early_stopping: dict = None,
                 gradient_clipping: dict = None):
        """Builds the PyTorch Lightning model.

        Args:
            obs_len: Length of the observed trajectory.
            pred_len: Length of the predicted trajectory.
            hidden_size: Hidden size of the LSTM.
            optimizer: Optimizer configuration.
            lr_scheduler: Learning rate scheduler configuration.
            early_stopping: Early stopping configuration.
            gradient_clipping: Gradient clipping configuration.
        """

        model = TrajectoryRNN(hidden_size=hidden_size,
                              encoder_arch=encoder_arch,
                              encoder_weights_init=encoder_weights_init,
                              decoder_arch=decoder_arch,
                              decoder_weights_init=decoder_weights_init,
                              pred_len=pred_len)

        super().__init__(
            model=model,
            obs_len=obs_len,
            pred_len=pred_len,
            num_samples=1,
            social_nce_loss_weight=0,
            social_nce_temperature=None,
            social_nce_proj_size=None,
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

        output = self.model(scene_SO2=scene_SO2)

        if self.training:
            # preds, history_encoding, social_encoding, map_encoding, map_patch
            return output + (None, None, None)
        else:
            return output

    def sampling_info(self) -> SamplingInfo | None:
        return None
