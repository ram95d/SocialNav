"""Model 6"""

from typing import Literal, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from model.modules import (
    TrajectoryTokenizer,
    TrajectorySampler,
    PositionalEncoding,
)
from .social_nce import ISocialNceCompatible
from .map_nce import IMapNceCompatible
import model.model_utils as model_utils
from model.pl_traj_model import BaseTrajectoryLitModule
from model.sampling_info import SamplingInfo




class MyTrajectoryModel7(nn.Module):

    # TODO: fix parametrization

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 vocab_size: int,
                 vocab_path: str,
                 num_layers: int,
                 hidden_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool,
                 activation):

        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        # TODO: likely can be precomputed.
        self.tokenizer = TrajectoryTokenizer(vocab_size=vocab_size,
                                             vocab_path=vocab_path)
        self.sampler = TrajectorySampler()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Max length of positional encoding is the obs_len + pred_len - 2,
        # since the transformer will be applied to the "relative" trajectory,
        # without the last observation. The relative trajectory has length
        # obs_len + pred_len - 1, so when removing the last observation
        # it has length obs_len + pred_len - 2.
        # The last observation is removed because it's the target to predict.
        traj_len = obs_len + pred_len
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      traj_len - 2)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                traj_tokens_BT: torch.Tensor,
                ) -> torch.Tensor:
        """Autoregressively predicts the future trajectory.

        Args:
            traj_BO2: The observed trajectory tensor.
                Shape: (batch_size, obs_len, 2).
            traj_BP2: The ground truth future trajectory tensor.
                Shape: (batch_size, pred_len, 2).

        Returns:
            The predicted logits tensor.
            Shape: (batch_size, pred_len, vocab_size).
        """

        batch_size, _ = traj_tokens_BT.shape

        # Embed trajectory tokens.
        traj_embedding_BTC = self.embedding(traj_tokens_BT)

        # Add positional encoding.
        traj_embedding_BTC = self.positional_encoding(traj_embedding_BTC)

        # Causal mask.
        attn_mask = nn.Transformer.generate_square_subsequent_mask(traj_embedding_BTC.size(1),
                                                                   device=traj_embedding_BTC.device)

        # Forward through transformer.
        output_BTC = self.transformer(traj_embedding_BTC, attn_mask, is_causal=True)

        # Predict logits.
        logits_BTA = self.output(output_BTC)

        return logits_BTA

    def predict(self,
                traj_tokens_BT: torch.Tensor,
                first_pos_B2: torch.Tensor,
                num_samples: int,
                temperature: float = 1.0,
                top_p: float = 0.0,
                top_k: int = 0,
                min_p: float = 0.0,
                ) -> torch.Tensor:
        """Autoregressively predicts the future trajectory.

        Args:
            traj_tokens_BT: The observed trajectory tensor.
                Shape: (batch_size, obs_len).
            num_samples: The number of samples to generate.
        """

        # TODO: pass a sampler config (temp, top_k, top_p, ...)

        # Z = B * K

        # Compute the logits for first token.
        logits_BTA = self.forward(traj_tokens_BT)

        # Sample the next token.
        next_token_BK = self.sampler.forward(logits_BA=logits_BTA[:, -1],
                                             num_samples=num_samples,
                                             temperature=temperature,
                                             top_p=top_p,
                                             top_k=top_k,
                                             min_p=min_p)

        # Flatten next token.
        next_token_Z1 = next_token_BK.view(-1, 1)

        # Expand the observed trajectory.
        traj_tokens_BKT = traj_tokens_BT.unsqueeze(1).expand(-1, num_samples, -1)
        traj_tokens_ZT = traj_tokens_BKT.reshape(-1, traj_tokens_BKT.size(2))

        # Concatenate the next token.
        traj_tokens_ZT = torch.cat([traj_tokens_ZT, next_token_Z1], dim=-1)

        # Autoregressively predict the other tokens.
        for _ in range(self.pred_len - 1):
            logits_ZTA = self.forward(traj_tokens_ZT)

            # Sample the next token.
            next_token_Z1 = self.sampler.forward(logits_BA=logits_ZTA[:, -1],
                                                 num_samples=1,
                                                 temperature=temperature,
                                                 top_p=top_p,
                                                 top_k=top_k,
                                                 min_p=min_p)

            # Concatenate the next token.
            traj_tokens_ZT = torch.cat([traj_tokens_ZT, next_token_Z1], dim=-1)

        # Decode the trajectory tokens.
        # TODO: decode from last observation.
        first_pos_Z2 = first_pos_B2.repeat_interleave(num_samples, dim=0)
        traj_hat_ZT2 = self.tokenizer.decode(traj_tokens_ZT, first_pos_Z2)
        traj_pred_hat_ZP2 = traj_hat_ZT2[:, self.obs_len:]

        # Reshape the predicted trajectory.
        traj_pred_hat_BKP2 = traj_pred_hat_ZP2.view(-1, num_samples, self.pred_len, 2)

        return traj_pred_hat_BKP2


class Model7LitModule(BaseTrajectoryLitModule):
    """PyTorch Lightning model for trajectory prediction."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 vocab_size: int,
                 vocab_path: str,
                 hidden_size: int,
                 num_layers: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool,
                 activation: str,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 min_p: float,
                 label_smoothing: float,
                 custom_label_smoothing_temperature: float,

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

        model = MyTrajectoryModel7(
            obs_len=obs_len,
            pred_len=pred_len,
            vocab_size=vocab_size,
            vocab_path=vocab_path,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            activation=activation,
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
            goal_net_pretrain_epochs=0,
            goal_loss_weight=0,
            goal_matching_loss_weight=0,
            goal_matching_loss_mode='all',
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
            gradient_clipping=gradient_clipping
        )

        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p

        # Compute cluster compatibility.
        if custom_label_smoothing_temperature > 0:
            cluster_centers_A2 = torch.load(vocab_path)
            dist_AA2 = cluster_centers_A2.unsqueeze(0) - cluster_centers_A2.unsqueeze(1)
            dist_AA = dist_AA2.norm(dim=-1)
            comp_AA = F.softmax(-dist_AA / custom_label_smoothing_temperature, dim=-1)
        else:
            comp_AA = torch.eye(vocab_size)

        self.register_buffer('cluster_comp_AA', comp_AA)

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
                ) -> dict:
        if traj_gt_BP2 is not None:
            # Compute the full trajectory.
            traj_BT2 = torch.cat([traj_BO2, traj_gt_BP2], dim=1)

            # Encode the full trajectory.
            traj_tokens_BT = self.model.tokenizer.encode(traj_BT2)

            # Input: full trajectory without the last token.
            input_BT = traj_tokens_BT[:, :-1]
            # Output: full trajectory without the first token.
            output_BT = traj_tokens_BT[:, 1:]

            # Predict the logits.
            logits_BTA = self.model.forward(input_BT)

            return {
                'logits_BTA': logits_BTA,
                'gt_tokens_BT': output_BT,
            }

        else:
            input_BT2 = traj_BO2

            # Encode the observed trajectory.
            traj_tokens_BT = self.model.tokenizer.encode(input_BT2)

            # Predict the future trajectory.
            traj_pred_hat_BKP2 = self.model.predict(traj_tokens_BT,
                                                    first_pos_B2=traj_BO2[:, 0],
                                                    num_samples=num_samples,
                                                    temperature=self.temperature,
                                                    top_p=self.top_p,
                                                    top_k=self.top_k,
                                                    min_p=self.min_p)

            # TODO: likely return the probabilities as well.
            return {
                'traj_pred_hat_BKP2': traj_pred_hat_BKP2,
            }

    def loss(self,
             output: dict,
             traj_obs_BO2: Tensor,
             traj_pred_BP2: Tensor) -> Tensor:

        if 'logits_BTA' not in output:
            return torch.tensor(0.0, device=self.device)

        logits_BTA = output['logits_BTA']
        gt_tokens_BT = output['gt_tokens_BT']

        # Flatten the logits.
        logits_ZA = logits_BTA.view(-1, logits_BTA.size(2))
        # Flatten the ground truth tokens.
        gt_tokens_Z = gt_tokens_BT.flatten()

        # Get the cluster compatibility (probabilities) (with potential smoothing).
        prob_ZA = self.cluster_comp_AA[gt_tokens_Z]

        # Compute the loss.
        # TODO: try label smoothing.
        loss = F.cross_entropy(logits_ZA,
                               # gt_tokens_Z,
                               prob_ZA,
                               label_smoothing=self.label_smoothing)

        return loss


    def sampling_info(self) -> SamplingInfo:
        return self.sampling_info_
