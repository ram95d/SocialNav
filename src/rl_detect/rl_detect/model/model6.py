"""Model 6"""

from typing import Literal, Optional

import torch
from torch import nn
from torch import Tensor

from model.modules import (
    TrajectoryEncoderTransformer2,
    TrajectoryDecoderTransformer3,
)
from .social_nce import ISocialNceCompatible
from .map_nce import IMapNceCompatible
import model.model_utils as model_utils
from model.pl_traj_model import BaseTrajectoryLitModule
from model.sampling_info import SamplingInfo

from model.goal.simple_goal_net import Simple3GoalNet


class MyTrajectoryModel6(nn.Module, ISocialNceCompatible, IMapNceCompatible):

    # TODO: fix parametrization

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 encoder_nhead: int,
                 encoder_dim_feedforward: int,
                 encoder_dropout: float,
                 encoder_norm_first: bool,
                 encoder_activation: str,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 decoder_nhead: int,
                 decoder_dim_feedforward: int,
                 decoder_dropout: float,
                 decoder_norm_first: bool,
                 decoder_activation: str):

        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.num_samples = num_samples


        self.goal_net = Simple3GoalNet(obs_len=obs_len,
                                       num_samples=num_samples,
                                       layer_sizes=[56, 40])

        # self.goal_net = MapGoalNet(num_samples=num_samples,
        #                            obs_len=obs_len)
        self.goal_embedding = nn.Linear(2, encoder_hidden_size)

        self.encoder = TrajectoryEncoderTransformer2(
            obs_len=obs_len,
            num_layers=encoder_num_layers,
            hidden_size=encoder_hidden_size,
            nhead=encoder_nhead,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            norm_first=encoder_norm_first,
            activation=encoder_activation
        )

        # TODO: reorder parameters
        self.decoder = TrajectoryDecoderTransformer3(
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            nhead=decoder_nhead,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
            norm_first=decoder_norm_first,
            activation=decoder_activation,
            pred_len=pred_len,
            num_samples=num_samples
        )


    def forward(self,
                traj_BO2: torch.Tensor,
                ) -> dict:

        # Z = B * K

        # Encode the trajectory.
        encoding_BoH = self.encoder(traj_BO2)

        # Compute goals.
        # traj_rel_Bo2 = traj_BO2.diff(1, dim=1)
        goal_BK2 = self.goal_net(traj_BO2)
        goal_delta_BK2 = goal_BK2 - traj_BO2[:, -1:, :]
        goal_delta_BKH = self.goal_embedding(goal_delta_BK2)

        # Expand trajectory emmbeddings.
        encoding_BKoH = encoding_BoH.unsqueeze(1).expand(-1, self.num_samples, -1, -1)

        # Concatenate trajectory and goal embeddings.
        goal_delta_detached_BKH = goal_delta_BKH.detach()
        encoding_BKOH = torch.cat([encoding_BKoH, goal_delta_detached_BKH[:, :, None]], dim=2)

        # Reshape the encoding.
        encoding_ZOH = encoding_BKOH.view(-1, encoding_BKOH.size(2), encoding_BKOH.size(3))

        # Decode the trajectory.
        last_pos_Z2 = traj_BO2[:, -1].repeat_interleave(self.num_samples, dim=0)
        output_ZP2 = self.decoder(encoding_ZOH, last_pos_Z2)

        # Compute absolute goals.
        # last_pos_B12 = traj_BO2[:, None, -1, :]
        # goal_BK2 = last_pos_B12 + goal_delta_BK2

        # Reshape the output.
        output_BKP2 = output_ZP2.view(-1, self.num_samples, self.pred_len, 2)

        return {
            'traj_pred_hat_BKP2': output_BKP2,
            'goal_BK2': goal_BK2,
            'history_embedding_BH': encoding_BoH
        }

    def social_encoding_size(self) -> int:
        return None

    def map_encoding_size(self) -> int:
        return None

class Model6LitModule(BaseTrajectoryLitModule):
    """PyTorch Lightning model for trajectory prediction."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 goal_net_pretrain_epochs: int,
                 goal_loss_weight: float,
                 goal_matching_loss_weight: float,
                 goal_matching_loss_mode: Literal['best', 'all'],
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 encoder_nhead: int,
                 encoder_dim_feedforward: int,
                 encoder_dropout: float,
                 encoder_norm_first: bool,
                 encoder_activation: str,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 decoder_nhead: int,
                 decoder_dim_feedforward: int,
                 decoder_dropout: float,
                 decoder_norm_first: bool,
                 decoder_activation: str,

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

        model = MyTrajectoryModel6(
            obs_len=obs_len,
            pred_len=pred_len,
            num_samples=num_samples,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            encoder_nhead=encoder_nhead,
            encoder_dim_feedforward=encoder_dim_feedforward,
            encoder_dropout=encoder_dropout,
            encoder_norm_first=encoder_norm_first,
            encoder_activation=encoder_activation,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_layers=decoder_num_layers,
            decoder_nhead=decoder_nhead,
            decoder_dim_feedforward=decoder_dim_feedforward,
            decoder_dropout=decoder_dropout,
            decoder_norm_first=decoder_norm_first,
            decoder_activation=decoder_activation
        )

        super().__init__(
            model=model,
            obs_len=obs_len,
            pred_len=pred_len,
            num_samples=num_samples,
            social_nce_loss_weight=0,
            social_nce_temperature=0,
            social_nce_proj_size=0,
            map_nce_loss_weight=0,
            map_nce_num_contour_points=0,
            map_nce_temperature=0,
            map_nce_proj_size=0,
            env_collision_loss_weight=0,
            goal_net_pretrain_epochs=goal_net_pretrain_epochs,
            goal_loss_weight=goal_loss_weight,
            goal_matching_loss_weight=goal_matching_loss_weight,
            goal_matching_loss_mode=goal_matching_loss_mode,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            gradient_clipping=gradient_clipping
        )

        self.sampling_info_ = SamplingInfo(
            noise_dim=0,
            noise_distrib='gaussian'
        )

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
                ) -> torch.Tensor \
                   | tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:

            return self.model(traj_BO2=traj_BO2)

    def loss(self,
             output: dict,
             traj_obs_BO2: Tensor,
             traj_pred_BP2: Tensor) -> Tensor:
        return torch.tensor(0.0, device=traj_obs_BO2.device)

    def sampling_info(self) -> SamplingInfo:
        return self.sampling_info_
