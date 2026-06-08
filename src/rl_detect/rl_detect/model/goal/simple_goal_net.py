"""Simple Goal sampling network."""

from typing import Optional
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.goal.pl_goal_model import BaseGoalNetLitModule
import rl_detect.utils as utils
import model.model_utils as model_utils
from model.modules import TrajectoryEncoderRNN1, PositionalEncoding
from model import metrics


class SimpleGoalNet(nn.Module):
    def __init__(self, num_samples: int, input_size: int):
        super().__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = 2

        self.net = nn.Sequential(
            nn.Linear(input_size * num_samples, input_size * num_samples * 2),
            nn.ReLU(),
            nn.Linear(input_size * num_samples * 2, self.output_size * num_samples)
        )

    def forward(self, past_emb_BC: Tensor):
        # Expand past embeddings to (B, K, C).
        # (B, C) -> (B, K, C)
        past_emb_BKC = past_emb_BC.unsqueeze(1).expand(-1, self.num_samples, -1)

        # Flatten to (B, K, C) -> (B, D = K*C)
        past_emb_BD = past_emb_BKC.flatten(start_dim=1)

        # Forward pass.
        future_emb_BD = self.net(past_emb_BD)

        # Reshape to (B, K, C).
        future_emb_BK2 = future_emb_BD.view(-1, self.num_samples, self.output_size)

        return future_emb_BK2


class Simple2GoalNet(nn.Module):
    def __init__(self, num_samples: int, input_size: int):
        super().__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = 2

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * num_samples * 2),
            nn.ReLU(),
            nn.Linear(input_size * num_samples * 2, self.output_size * num_samples)
        )

    def forward(self, past_emb_BC: Tensor):
        # Forward pass.
        future_emb_BD = self.net(past_emb_BC)

        # Reshape to (B, K, C).
        future_emb_BK2 = future_emb_BD.view(-1, self.num_samples, self.output_size)

        return future_emb_BK2




class SimpleSamplingGoalNet(nn.Module):
    def __init__(self, num_samples: int, input_size: int):
        super().__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = 2
        self.noise_dim = input_size // 2

        self.net = nn.Sequential(
            nn.Linear(input_size + self.noise_dim, input_size * 16),
            nn.ReLU(),
            nn.Linear(input_size * 16, self.output_size)
        )

    def forward(self, past_emb_BC: Tensor):
        # Sample noise.
        batch_size = past_emb_BC.size(0)
        noise_shape = (batch_size, self.num_samples, self.noise_dim)
        noise_BKC = torch.randn(noise_shape, device=past_emb_BC.device)

        # Expand past embeddings to (B, K, C).
        # (B, C) -> (B, K, C)
        past_emb_BKC = past_emb_BC.unsqueeze(1).expand(-1, self.num_samples, -1)

        # Concatenate noise to past embeddings.
        past_emb_noise_BKC = torch.cat([past_emb_BKC, noise_BKC], dim=-1)

        # Flatten to batch dimension (B, K, C) -> (B*K, C).
        past_emb_noise_bC = past_emb_noise_BKC.view(-1, self.input_size + self.noise_dim)

        # Forward pass.
        goal_b2 = self.net(past_emb_noise_bC)

        # Reshape to (B, K, C).
        goal_BK2 = goal_b2.view(-1, self.num_samples, self.output_size)

        return goal_BK2


class SimpleTransformerGoalNet(nn.Module):
    def __init__(self, num_samples: int, input_size: int):
        super().__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = 2

        transformer_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                       nhead=2,
                                                       dim_feedforward=input_size * 2,
                                                       dropout=0.1,
                                                       activation='relu',
                                                       norm_first=True,
                                                       batch_first=True)

        self.pos_encoder = PositionalEncoding(input_size, dropout=0.0, max_len=self.num_samples)
        self.transformer = nn.TransformerEncoder(transformer_layer,
                                                 num_layers=2)

        self.output_layer = nn.Linear(input_size, self.output_size)





    def forward(self, past_emb_BC: Tensor):
        # Expand past embeddings to (B, K, C).
        # (B, C) -> (B, K, C)
        past_emb_BKC = past_emb_BC.unsqueeze(1).expand(-1, self.num_samples, -1)

        # Forward pass.
        past_emb_BKC = self.pos_encoder(past_emb_BKC)
        future_emb_BKC = self.transformer(past_emb_BKC)

        # Output layer.
        future_emb_BK2 = self.output_layer(future_emb_BKC)

        return future_emb_BK2


################################################################################
################################################################################
################################################################################
################################################################################

# Actual implementation of the goal net.


class Simple3GoalNet(nn.Module):
    def __init__(self, obs_len: int, num_samples: int, layer_sizes: list[int]):
        super().__init__()

        if len(layer_sizes) <= 0:
            raise ValueError("layer_sizes must have at least 1 element.")

        self.coord_dim = 2
        self.num_samples = num_samples

        # Input size is the flattened relative displacements.
        input_size = (obs_len - 1) * self.coord_dim

        # Build layers
        layers = []
        current_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU())
            current_size = size
        # Output layer.
        layers.append(nn.Linear(current_size, self.coord_dim * num_samples))

        self.net = nn.Sequential(*layers)

    def forward(self, traj_BO2: Tensor):
        # Compute relative displacements.
        traj_rel_BO2 = traj_BO2.diff(1, dim=1)

        # Flatten (B, O, 2) -> (B, C=O*2).
        traj_BC = traj_rel_BO2.flatten(start_dim=1)

        # Forward pass.
        goal_BD = self.net(traj_BC)

        # Reshape to (B, K, C).
        goal_BK2 = goal_BD.view(-1, self.num_samples, self.coord_dim)

        # Compute absolute goals.
        goal_BK2 = goal_BK2 + traj_BO2[:, -1:]

        return goal_BK2


class SimpleGoalNetLitModule(BaseGoalNetLitModule):
    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 layer_sizes: list[int],
                 optimizer: Optional[dict] = None,
                 lr_scheduler: Optional[dict] = None,
                 early_stopping: Optional[dict] = None,
                 gradient_clipping: Optional[dict] = None):

        model = Simple3GoalNet(
            obs_len=obs_len,
            num_samples=num_samples,
            layer_sizes=layer_sizes,
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

        self.save_hyperparameters()

    def forward(self,
                traj_BO2: Tensor,
                map_mask_B1HW: Tensor,
                hom_meters2mask_B33: Tensor,
                hom_mask2meters_B33: Tensor
                ) -> dict:

        goal_BK2 = self.model(traj_BO2)

        return {
            "goal_BK2": goal_BK2,
        }

    def loss(self, output: dict, gt_goal_B2: Tensor, traj_BO2: Tensor, hom_meters2mask_B33) -> Tensor:
        goal_BK2 = output["goal_BK2"]

        # Compute the loss.
        loss = self._best_goal_loss(goal_BK2, gt_goal_B2)

        return loss
