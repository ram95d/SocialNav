import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialNceLoss(nn.Module):
    """Implementation for the Social NCE loss."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 query_embedder: nn.Module,
                 key_embedder: nn.Module,
                 temperature: float):
        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.temperature = temperature

        self.query_embedder = query_embedder
        self.key_embedder = key_embedder

    def forward(self, scene_BA2, social_embeddings_BH, scene_idx_B):
        """Compute the Social NCE loss.

        Args:
            scene_BA2: The batched scene tensor (Batch, Time, XY).
            social_embeddings_BH: The batched social embeddings (Batch, Hidden).
            scene_idx_B: Tensor of scene indices for each trajectory (Batch,).

        Returns:
            The Social NCE loss averaged over all scenes.
        """
        device = scene_BA2.device

        # Get scene sizes (S: number of people in each scene)
        unique_scenes_U, scene_counts_U = torch.unique(scene_idx_B, return_counts=True)

        # Skip computation if no valid scenes (scenes with > 1 person)
        valid_scenes_mask_U = scene_counts_U > 1
        if not valid_scenes_mask_U.any():
            return torch.tensor(0., device=device)

        # Create a mask for valid trajectories (those in scenes with > 1 person)
        valid_scenes_V = unique_scenes_U[valid_scenes_mask_U]
        valid_traj_mask_B = torch.zeros_like(scene_idx_B, dtype=torch.bool)
        for scene_idx in valid_scenes_V:
            valid_traj_mask_B |= (scene_idx_B == scene_idx)

        # Filter out trajectories from invalid scenes
        # After filtering, B becomes the total number of trajectories in valid scenes
        scene_BA2 = scene_BA2[valid_traj_mask_B]
        social_embeddings_BH = social_embeddings_BH[valid_traj_mask_B]
        scene_idx_B = scene_idx_B[valid_traj_mask_B]

        # Sample positive and negative samples
        pos_samples_B2, neg_samples_BN2, neg_mask_BN = self._social_sampling_batched(
            scene_BA2, scene_idx_B)

        # Compute query and keys (C: projection size)
        query_BC = self.query_embedder(social_embeddings_BH)
        pos_key_BC = self.key_embedder(pos_samples_B2)
        neg_key_BNC = self.key_embedder(neg_samples_BN2)

        # Normalize vectors
        query_BC = F.normalize(query_BC, dim=-1)
        pos_key_BC = F.normalize(pos_key_BC, dim=-1)
        neg_key_BNC = F.normalize(neg_key_BNC, dim=-1)

        # Compute logits
        pos_logits_B = torch.einsum('bc,bc->b', query_BC, pos_key_BC) / self.temperature
        neg_logits_BN = torch.einsum('bc,bnc->bn', query_BC, neg_key_BNC) / self.temperature

        # Apply mask to invalid negatives (set logits to -inf)
        neg_logits_BN = neg_logits_BN * neg_mask_BN - (1 - neg_mask_BN) * 1e9

        # Compute loss
        logits_BL = torch.cat((pos_logits_B.unsqueeze(1), neg_logits_BN), dim=1)  # L: 1 + num_neg

        # Create scene-wise weights to average losses within each scene
        scene_weights_B = torch.zeros_like(scene_idx_B, dtype=torch.float)
        for scene_idx in valid_scenes_V:
            scene_mask_B = (scene_idx_B == scene_idx)
            scene_weights_B[scene_mask_B] = 1.0 / scene_mask_B.sum()

        # Compute weighted cross entropy loss
        labels_B = torch.zeros(logits_BL.size(0), dtype=torch.long, device=device)
        losses_B = F.cross_entropy(logits_BL, labels_B, reduction='none')
        weighted_loss = (losses_B * scene_weights_B).sum()

        return weighted_loss

    @torch.no_grad()
    def _social_sampling_batched(self, scene_BA2, scene_idx_B):
        """Vectorized social sampling handling multiple scenes with padding."""
        batch_size = scene_BA2.size(0)
        device = scene_BA2.device

        # Sample delta_t
        delta_t = torch.randint(1, self.pred_len, (1,)).item()
        time = self.obs_len + delta_t

        # Constants
        discomfort_area = 0.35
        noise_scale = 0.05
        num_neg_samples = 8  # n
        delta_degree = 2 * math.pi / num_neg_samples

        # Get positions
        fut_positions_B2 = scene_BA2[:, time]
        curr_positions_B2 = scene_BA2[:, self.obs_len]

        # Generate angles for negative samples
        angles_n = torch.arange(0, 2 * math.pi, delta_degree, device=device)
        deltas_n2 = discomfort_area * torch.stack(
            (torch.cos(angles_n), torch.sin(angles_n)), dim=1)

        # Generate base negative samples (B, n, 2)
        # n: number of negative samples.
        neg_samples_Bn2 = fut_positions_B2.unsqueeze(1) + deltas_n2
        neg_samples_Bn2 = neg_samples_Bn2 + noise_scale * torch.randn_like(neg_samples_Bn2)

        # Get scene information
        unique_scenes_U, scene_counts_U = torch.unique(scene_idx_B, return_counts=True)
        max_scene_size = scene_counts_U.max().item()
        max_neg_samples = (max_scene_size - 1) * num_neg_samples if max_scene_size > 1 else 0 # -1 for excluding self

        all_neg_samples = []
        all_masks = []

        for scene_idx, scene_size in zip(unique_scenes_U, scene_counts_U):
            scene_size = scene_size.item()
            if scene_size == 1:
                continue  # Skip scenes with single pedestrian

            scene_mask_B = (scene_idx_B == scene_idx)
            curr_num_neg_samples = (scene_size - 1) * num_neg_samples

            # Get scene negatives (S, n, 2)
            scene_negs_Sn2 = neg_samples_Bn2[scene_mask_B]

            # Create all combinations (S, S, n, 2)
            scene_negs_SSn2 = scene_negs_Sn2.unsqueeze(0).repeat(scene_size, 1, 1, 1)

            # Exclude self-samples and reshape (S, (S-1)*n, 2)
            exclude_mask_SS = ~torch.eye(scene_size, dtype=torch.bool, device=device)
            scene_negs_SN2 = scene_negs_SSn2[exclude_mask_SS].view(scene_size, -1, 2)

            # Pad to max_N and create mask
            padded_neg = torch.zeros((scene_size, max_neg_samples, 2), device=device)
            padded_neg[:, :curr_num_neg_samples] = scene_negs_SN2

            scene_mask = torch.zeros((scene_size, max_neg_samples), device=device)
            scene_mask[:, :curr_num_neg_samples] = 1.0

            all_neg_samples.append(padded_neg)
            all_masks.append(scene_mask)

        # Handle case where all scenes have only 1 pedestrian
        if not all_neg_samples:
            return (torch.zeros_like(fut_positions_B2),
                    torch.zeros((batch_size, 0, 2), device=device),
                    torch.zeros((batch_size, 0), device=device))

        neg_samples_BN2 = torch.cat(all_neg_samples, dim=0)
        neg_mask_BN = torch.cat(all_masks, dim=0)

        # Generate positive samples
        pos_samples_B2 = fut_positions_B2 + noise_scale * torch.randn_like(fut_positions_B2)

        # Convert to relative coordinates
        pos_samples_B2 = curr_positions_B2 - pos_samples_B2
        neg_samples_BN2 = curr_positions_B2.unsqueeze(1) - neg_samples_BN2

        return pos_samples_B2, neg_samples_BN2, neg_mask_BN


class ISocialNceCompatible(ABC):
    """Interface for models compatible with SocialNCE loss."""

    @abstractmethod
    def social_encoding_size(self) -> int:
        pass


class SocialQueryEmbedder(nn.Module):
    """Query embedder for Social NCE loss."""

    def __init__(self, traj_social_encoding_size: int, proj_size: int):
        super().__init__()

        self.net = nn.Linear(traj_social_encoding_size, proj_size)

    def forward(self, traj_social_embeddings_SH):
        return self.net(traj_social_embeddings_SH)


class SocialKeyEmbedder(nn.Module):
    """Key embedder for Social NCE loss."""

    def __init__(self, input_size: int, proj_size: int):
        """Initialize the key embedder.

        Args:
            input_size: The size of the input (positive or negative samples).
            proj_size: The dimensionality of the space on which the samples
                are projected.
        """

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, proj_size * 2),
            nn.ReLU(),
            nn.Linear(proj_size * 2, proj_size),
        )

    def forward(self, key):
        return self.net(key)
