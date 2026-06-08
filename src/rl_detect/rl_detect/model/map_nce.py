import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv


# TODO:
# - try other query and key embedders.


class MapNceLoss(nn.Module):
    """Implementation for the Map NCE loss."""

    def __init__(self,
                 obs_len: int,
                 pred_len: int,
                 num_contour_points: int,
                 query_embedder: nn.Module,
                 key_embedder: nn.Module,
                 temperature: float):
        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_contour_points = num_contour_points
        self.temperature = temperature

        self.query_embedder = query_embedder
        self.key_embedder = key_embedder

    def forward(self,
                scene_BA2: torch.Tensor,
                traj_map_embedding_BH: torch.Tensor,
                map_patch_B1HW: torch.Tensor) -> torch.Tensor:
        """Compute the Map NCE loss.

        Args:
            scene_BA2: The batched scene tensor.
                Shape: (batch_size, agents_per_scene, 2).
            traj_map_embedding_BH: Trajectory and map embeddings tensor
                (combined in some way).
                Shape: (batch_size, traj_map_encoding_size).
            map_patch_B1HW: The map patches tensor.
                Shape: (batch_size, 1, height, width).

        Returns:
            The Map NCE loss.
        """
        # Sample positive and negative samples.
        pos_samples_B2, neg_samples_BN2, valid_B = \
            self._map_sampling(scene_BA2, map_patch_B1HW)

        # Compute the query and the keys.
        query_BC = self.query_embedder(traj_map_embedding_BH)
        pos_key_BC = self.key_embedder(pos_samples_B2)
        neg_key_BNC = self.key_embedder(neg_samples_BN2)

        # Normalize the vectors.
        query_BC = F.normalize(query_BC, dim=-1)
        pos_key_BC = F.normalize(pos_key_BC, dim=-1)
        neg_key_BNC = F.normalize(neg_key_BNC, dim=-1)

        temp = self.temperature
        # Compute the logits.
        pos_logits_B = torch.einsum('bc,bc->b', query_BC, pos_key_BC) / temp
        neg_logits_BN = torch.einsum('bc,bnc->bn', query_BC, neg_key_BNC) / temp

        # Compute the loss.
        # Shape: (batch_size, num_neg_samples + 1).
        logits = torch.cat((pos_logits_B.unsqueeze(1), neg_logits_BN), dim=1)

        # For each person, the "correct class" (positive sample) is the first.
        if valid_B.any():
            loss = F.cross_entropy(logits[valid_B],
                                torch.zeros(valid_B.sum(),
                                        dtype=torch.long,
                                        device=logits.device))
        else:
            # No valid contour points.
            loss = torch.tensor(0., device=logits.device)

        return loss

    @torch.no_grad()
    def _map_sampling(self, scene_BA2, map_patch_B1HW):
        """Sample positive and negative samples for the Map NCE loss."""

        # F: number of countour points extracted for each map mask.
        # n: number of negative samples around each selected contour point.
        # N: total number of negative samples (for each person).
        # N = self.num_contour_points * n, where n = num_neg_samples = 8.

        batch_size = scene_BA2.size(0)

        # Extract F contour points from the map mask of each person.
        contour_seed_BF2, valid_B = \
            self._extract_contour_points(map_patch_B1HW,
                                        self.num_contour_points)

        # Delta t.
        # TODO: maybe parameterize this.
        # delta_t = 3
        delta_t = torch.randint(4, self.pred_len, (1,)).item()

        # Time.
        time = self.obs_len + delta_t

        # Discomfort area (meters).
        discomfort_area = 0.5

        # Noise scale (meters).
        noise_scale = 0.3

        # Number of negative samples (around each contour point).
        num_neg_samples = 8

        # Delta degree of the circle.
        delta_degree = 2 * math.pi / num_neg_samples

        # Get the future positions at the given time.
        fut_positions_B2 = scene_BA2[:, time]

        # Angles.
        angles_n = torch.arange(0, 2 * math.pi, delta_degree)

        # Deltas wrt the position of the contour point.
        deltas_n2 = discomfort_area * torch.stack((torch.cos(angles_n),
                                                    torch.sin(angles_n)),
                                                   dim=1).to(scene_BA2)

        # Negative samples.
        # n negative samples around each countour point.
        neg_samples_BFn2 = contour_seed_BF2.unsqueeze(2) + deltas_n2

        # Add noise.
        noise_BFn2 = noise_scale * torch.randn_like(neg_samples_BFn2)
        neg_samples_BFn2 += noise_BFn2

        # Merge the negative samples of the same person.
        neg_samples_BN2 = neg_samples_BFn2.view(batch_size, -1, 2)

        # Sample 1 positive sample for each person.
        pos_samples_B2 = \
            fut_positions_B2 + noise_scale * torch.randn_like(fut_positions_B2)

        # Transform positive samples to patch pixel coordinates.

        # Current position of the person.
        curr_positions_B2 = scene_BA2[:, self.obs_len]
        prev_positions_B2 = scene_BA2[:, self.obs_len - 1]

        # Current direction vector of the person.
        equals_B = torch.isclose(curr_positions_B2, prev_positions_B2).\
                         all(dim=1)
        if equals_B.any():
            # If positions are the same, randomly perturb the prev.
            prev_positions_B2 = prev_positions_B2.clone()

            # Sample angles uniformly between 0 and 2pi for the perturbation.
            angles_B = 2 * torch.pi * torch.rand(equals_B.sum(),
                                                 device=scene_BA2.device)
            delta_B2 = torch.stack([torch.cos(angles_B),
                                            torch.sin(angles_B)], dim=-1)

            # Apply perturbations to the previous positions.
            prev_positions_B2[equals_B] += delta_B2

        curr_dir_B2 = curr_positions_B2 - prev_positions_B2

        # Normalize the direction vector.
        curr_dir_B2 /= torch.norm(curr_dir_B2, dim=-1, keepdim=True)

        # Direction vector from the current position to the positive sample.
        fut_dir_B2 = pos_samples_B2 - curr_positions_B2
        # Norm of the direction vector.
        fut_dir_norm_B = torch.norm(fut_dir_B2, dim=-1, keepdim=True)
        # Normalize the direction vector.
        fut_dir_B2 /= fut_dir_norm_B

        # TODO: test this code.
        # Straight direction.
        straight_dir_2 = torch.tensor([0., 1.], device=scene_BA2.device)
        # Angle between the current direction and the straight direction.
        angle_B = torch.acos(torch.sum(curr_dir_B2 * straight_dir_2, dim=-1))

        # Rotate the positive samples by the angle.

        # Create the rotation matrix.
        cos_vals = torch.cos(angle_B).reshape(-1, 1, 1)
        sin_vals = torch.sin(angle_B).reshape(-1, 1, 1)
        rotation_matrix_B22 = torch.cat([
            torch.cat([cos_vals, -sin_vals], dim=-1),
            torch.cat([sin_vals, cos_vals], dim=-1)
        ], dim=-2)

        # Calculate the rotated directions.
        rot_dir_B2 = torch.einsum('bij,bj->bi', rotation_matrix_B22, fut_dir_B2)

        # Scale by the future position distances.
        pos_samples_B2 = curr_positions_B2 + rot_dir_B2 * fut_dir_norm_B

        # Transform positive samples to patch pixel coordinates.

        # TODO: this code assumes that the patch size is 100x100,
        # that each pixel is 0.1 meters, and that the person is in the
        # (50, 10) pixel position in the patch.

        # Current position of the person in patch pixel coordinates.
        curr_pos_ppc_B2 = torch.tensor([50, 10], device=scene_BA2.device).\
                                repeat(batch_size, 1)
        # Positive samples in patch pixel coordinates.
        pos_samples_ppc_B2 = \
            (pos_samples_B2 - curr_positions_B2) / 0.1 + curr_pos_ppc_B2

        #######################################################
        # TODO: debug remove.

        # if valid_B.any():
        #     mask_patch = map_patch_B1HW[0, 0].cpu().numpy()
        #     mask_patch = cv.cvtColor(mask_patch, cv.COLOR_GRAY2BGR)
        #     cv.circle(mask_patch,
        #                 (int(pos_samples_ppc_B2[0, 0].item()),
        #                 int(pos_samples_ppc_B2[0, 1].item())),
        #                 1, (0, 255, 0), -1)
        #     for neg_sample in neg_samples_BN2[0]:
        #         cv.circle(mask_patch,
        #                 (int(neg_sample[0].item()),
        #                 int(neg_sample[1].item())),
        #                 1, (0, 0, 255), -1)
        #     cv.imshow('mask_patch', mask_patch)
        #     cv.waitKey(5000)

        #######################################################

        return pos_samples_ppc_B2, neg_samples_BN2, valid_B

    @torch.no_grad()
    def _extract_contour_points(self,
                            map_patch_B1HW: torch.Tensor,
                            n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the contour points of the map patches."""
        # Similar logic but for batched input.
        mask_patches_BHW = (map_patch_B1HW * 255).squeeze(1).\
                                                cpu().\
                                                numpy().\
                                                astype(np.uint8)

        contours_list = []
        valid_list = []
        for mask_patch_HW in mask_patches_BHW:
            # Invert the mask so that the contours are extracted correctly.
            mask_patch_HW = cv.bitwise_not(mask_patch_HW)
            # Extract the contours.
            contours, _ = cv.findContours(mask_patch_HW,
                                          cv.RETR_LIST,
                                          cv.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                # No contours found.
                contours_E2 = np.empty((0, 2), dtype=np.float32)
            else:
                contours_E2 = np.concatenate(contours).squeeze(1)

            # Note: to keep the loss computation parallel, we need to
            # keep the number of points (negative samples) the same for
            # each person.
            if contours_E2.shape[0] == 0:
                # Fill with zeros.
                # This will be masked at loss computation time.
                contours_E2 = np.zeros((n, 2), dtype=np.float32)
                valid = False

            else:
                # Sample n points from the contours with or without
                # replacement, depending on the number of points in the
                # contours.
                count = contours_E2.shape[0]
                indices = np.random.choice(count, n, replace=count < n)
                contours_E2 = contours_E2[indices].astype(np.float32)
                valid = True

            contours_list.append(contours_E2)
            valid_list.append(valid)

        contour_seed_BF2 = torch.from_numpy(np.stack(contours_list)).\
                                to(map_patch_B1HW.device)
        valid_B = torch.tensor(valid_list,
                            dtype=torch.bool,
                            device=map_patch_B1HW.device)

        return contour_seed_BF2, valid_B


class IMapNceCompatible(ABC):
    """Interface for models that are compatible with the MapNCE loss."""

    @abstractmethod
    def map_encoding_size(self) -> int:
        """Returns the size of the map encoding."""
        pass


class MapQueryEmbedder(nn.Module):
    """Query embedder for the Map NCE loss."""

    def __init__(self, traj_map_encoding_size: int, proj_size: int):
        super().__init__()

        self.net = nn.Linear(traj_map_encoding_size, proj_size)

    def forward(self, traj_map_embedding_SH):
        return self.net(traj_map_embedding_SH)


class MapKeyEmbedder(nn.Module):
    """Key embedder for the Map NCE loss."""

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
