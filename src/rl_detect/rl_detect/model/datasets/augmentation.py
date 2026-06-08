"""Data augmentation for trajectory data:
- shift
- rotation
- noise
- flip
- centering

The transformations preserve the relative positions of the pedestrians in the
scene, up to some noise (added by the noise transformation).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

class TrajectoryAugmentation(nn.Module):
    """Vectorized augmentation module for trajectory data."""

    def __init__(self,
                 center_ref_observation: int,
                 shift_prob: float,
                 rotate_prob: float,
                 flip_prob: float,
                 noise_prob: float,
                 noise_scale: float):
        super().__init__()
        self.center_ref_observation = center_ref_observation
        self.shift_prob = shift_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_scale = noise_scale

    @torch.no_grad()
    def forward(self, traj_bSA2: Tensor) -> tuple[Tensor, Tensor]:
        """Apply transformations to a batch of scenes in parallel.

        Args:
            scene_BSA2: Batch of scene tensors.
                Shape: (num_scenes, scene_size, obs_len + pred_len, 2)

        Returns:
            Transformed scenes and transformation matrices.
                Shape: (num_scenes, scene_size, obs_len + pred_len, 2),
                       (num_scenes, 3, 3)
        """

        num_scenes = traj_bSA2.shape[0]
        device = traj_bSA2.device

        # Center the scenes.
        if self.center_ref_observation is not None:
            traj_bSA2, barycenters_b2 = \
                center_scenes(traj_bSA2, self.center_ref_observation)
        else:
            barycenters_b2 = torch.zeros(num_scenes, 2, device=device)

        # Apply transformations.
        traj_bSA2, shift_matrix_b33 = shift_scenes(traj_bSA2, self.shift_prob)
        traj_bSA2, rot_matrix_b33 = rotate_scenes(traj_bSA2, self.rotate_prob)
        traj_bSA2, flip_matrix_b33 = flip_scenes(traj_bSA2, self.flip_prob)
        traj_bSA2 = add_noise(traj_bSA2,
                              noise_prob=self.noise_prob,
                              noise_scale=self.noise_scale)

        # Build transformation matrices for each scene.
        transform_matrix_b33 = torch.eye(3, device=device).unsqueeze(0).repeat(num_scenes, 1, 1)
        transform_matrix_b33[:, 0:2, 2] = -barycenters_b2
        transform_matrix_b33 = torch.bmm(shift_matrix_b33, transform_matrix_b33)
        transform_matrix_b33 = torch.bmm(rot_matrix_b33, transform_matrix_b33)
        transform_matrix_b33 = torch.bmm(flip_matrix_b33, transform_matrix_b33)

        inv_transform_matrix_b33 = torch.inverse(transform_matrix_b33)

        return traj_bSA2, transform_matrix_b33, inv_transform_matrix_b33


def shift_scenes(traj_bSA2: Tensor, shift_prob: float) -> tuple[Tensor, Tensor]:
    """Apply random shifts independently to each scene in the batch.

    Args:
        traj_bSA2: Batch of scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        shift_prob: Probability of applying shift to each scene

    Returns:
        Tuple:
        - Scene tensors with shifts applied independently to each scene.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        - Transformation matrices for the shifts.
            Shape: (num_scenes, 3, 3)
    """

    num_scenes = traj_bSA2.shape[0]
    device = traj_bSA2.device

    # Initialize transformation matrices as identity
    shift_matrix_b33 = torch.eye(3, device=device).unsqueeze(0).repeat(num_scenes, 1, 1)

    # Generate random probabilities for each scene.
    shift_mask_b = _generate_random_scene_mask(num_scenes, shift_prob, device)

    if not shift_mask_b.any():
        return traj_bSA2, shift_matrix_b33

    # Generate random shifts for each scene (-5 to 5 meters).
    shift_b2 = torch.randint(-5, 6, (num_scenes, 2), device=device).float()

    # Update transformation matrices for shifted scenes
    shift_matrix_b33[shift_mask_b, 0:2, 2] = shift_b2[shift_mask_b]

    # Apply shifts only to selected scenes.
    shift_mask_b111 = shift_mask_b.view(-1, 1, 1, 1)
    shift_b112 = shift_b2.view(num_scenes, 1, 1, 2)
    traj_bSA2 = torch.where(shift_mask_b111, traj_bSA2 + shift_b112, traj_bSA2)

    return traj_bSA2, shift_matrix_b33


def rotate_scenes(traj_bSA2: Tensor, rotate_prob: float) -> tuple[Tensor, Tensor]:
    """Apply random rotations independently to each scene in the batch.

    Args:
        traj_bSA2: Batch of scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        rotate_prob: Probability of applying rotation to each scene

    Returns:
        Tuple:
        - Scene tensors with rotations applied independently to each scene.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        - Transformation matrices for the rotations.
            Shape: (num_scenes, 3, 3)
    """

    num_scenes = traj_bSA2.shape[0]
    device = traj_bSA2.device

    # Initialize transformation matrices as identity
    rot_matrix_b33 = torch.eye(3, device=device).unsqueeze(0).repeat(num_scenes, 1, 1)

    # Generate random probabilities for each scene.
    rotate_mask_b = _generate_random_scene_mask(num_scenes, rotate_prob, device)

    if not rotate_mask_b.any():
        return traj_bSA2, rot_matrix_b33

    # Generate random angles for each scene.
    angle_b = torch.rand(num_scenes, device=device) * 2 * math.pi

    # Create rotation matrices for all scenes.
    cos_theta_b = torch.cos(angle_b)
    sin_theta_b = torch.sin(angle_b)
    rot_matrix_b22 = torch.stack([
        torch.stack([cos_theta_b, -sin_theta_b], dim=1),
        torch.stack([sin_theta_b, cos_theta_b], dim=1)
    ], dim=2)

    # Update transformation matrices for rotated scenes
    rot_matrix_b33[rotate_mask_b, 0:2, 0:2] = rot_matrix_b22[rotate_mask_b]

    # Apply rotations.
    rotated_traj_bSA2 = torch.einsum('bsai,bij->bsaj', traj_bSA2, rot_matrix_b22)

    # Apply rotations only to selected scenes.
    rotate_mask_b111 = rotate_mask_b.view(-1, 1, 1, 1)
    traj_bSA2 = torch.where(rotate_mask_b111, rotated_traj_bSA2, traj_bSA2)

    return traj_bSA2, rot_matrix_b33


def flip_scenes(traj_bSA2: Tensor, flip_prob: float) -> tuple[Tensor, Tensor]:
    """Apply random flips independently to each scene in the batch.

    Args:
        traj_bSA2: Batch of scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        flip_prob: Probability of applying flip to each scene

    Returns:
        Tuple:
        - Scene tensors with flips applied independently to each scene.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        - Transformation matrices for the flips.
            Shape: (num_scenes, 3, 3)
    """

    num_scenes = traj_bSA2.shape[0]
    device = traj_bSA2.device

    # Initialize transformation matrices as identity
    flip_matrix_b33 = torch.eye(3, device=device).unsqueeze(0).repeat(num_scenes, 1, 1)

    # Generate random probabilities for each scene.
    flip_mask_b = _generate_random_scene_mask(num_scenes, flip_prob, device)

    if not flip_mask_b.any():
        return traj_bSA2, flip_matrix_b33

    # Generate random flip types for each scene (0: x, 1: y, 2: both).
    flip_types_b = torch.randint(0, 4, (num_scenes,), device=device)

    # Create flip matrices.
    flip_x_b = (flip_types_b == 0) | (flip_types_b == 2)
    flip_y_b = (flip_types_b == 1) | (flip_types_b == 2)

    # Update transformation matrices for flipped scenes
    flip_matrix_b33[flip_mask_b & flip_x_b, 0, 0] = -1  # Flip x
    flip_matrix_b33[flip_mask_b & flip_y_b, 1, 1] = -1  # Flip y

    # Create flip vectors for trajectory transformation
    flip_matrix_b112 = torch.stack([
        flip_x_b * -2 + 1,  # Convert True/False to -1/1.
        flip_y_b * -2 + 1
    ], dim=1).view(num_scenes, 1, 1, 2)

    # Apply flips only to selected scenes.
    flip_mask_b111 = flip_mask_b.view(-1, 1, 1, 1)
    traj_bSA2 = torch.where(flip_mask_b111, traj_bSA2 * flip_matrix_b112, traj_bSA2)

    return traj_bSA2, flip_matrix_b33


def add_noise(traj_bSA2: Tensor,
              noise_prob: float,
              noise_scale: float) -> Tensor:
    """Add noise independently to each scene in the batch.

    Args:
        scene_bSA2: Batch of scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        noise_prob: Probability of applying noise to each scene
        noise_scale: Scale of the noise. 99% of noise is less than this value

    Returns:
        Scene tensors with noise applied independently to each scene.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
    """

    num_scenes = traj_bSA2.shape[0]
    device = traj_bSA2.device

    # Generate random probabilities for each scene.
    noise_mask_b = _generate_random_scene_mask(num_scenes, noise_prob, device)

    if not noise_mask_b.any():
        return traj_bSA2

    # Set standard deviation such that 99% of noise is less than scale.
    std_dev = noise_scale / 2.33

    # Generate noise for all scenes.
    noise_bSA2 = torch.randn_like(traj_bSA2) * std_dev

    # Where noise_mask is True, add noise; where False, keep original.
    noise_mask_b111 = noise_mask_b.view(-1, 1, 1, 1)
    traj_bSA2 = torch.where(noise_mask_b111, traj_bSA2 + noise_bSA2, traj_bSA2)

    return traj_bSA2


def center_scenes(traj_bSA2: Tensor,
                  ref_observation_idx: int) -> tuple[Tensor, Tensor]:
    """Center all scenes in the batch using the reference observation
    for each scene.

    Args:
        scene_BSA2: Batch of scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        ref_observation_idx: Index of the observation to use for centering.

    Returns:
        Tuple:
        - Centered scene tensors.
            Shape: (num_scenes, scene_size, obs_len + pred_len, 2)
        - Barycenters used for centering.
            Shape: (num_scenes, 2)
    """

    # Create mask for non-NaN positions at reference observation.
    not_nan_mask_bS = ~torch.isnan(traj_bSA2[:, :, ref_observation_idx, 0])

    # Get reference positions for each scene.
    ref_position_bS2 = traj_bSA2[:, :, ref_observation_idx]

    # Calculate barycenters (mean of non-NaN positions).
    # First create a masked version of positions where NaN values are replaced with 0
    masked_position_bS2 = ref_position_bS2 * not_nan_mask_bS[:, :, None]
    # Sum positions and divide by count of non-NaN values.
    barycenter_b2 = masked_position_bS2.sum(dim=1) / not_nan_mask_bS.sum(dim=1, keepdim=True)

    # Center all scenes using broadcasting.
    centered_traj_bSA2 = traj_bSA2 - barycenter_b2[:, None, None]

    return centered_traj_bSA2, barycenter_b2


def undo_center_scenes_batch(scene_BSA2: Tensor,
                           barycenters: Tensor) -> Tensor:
    """Undo the centering transformation for all scenes in parallel.

    Args:
        scene_BSA2: Batch of centered scene tensors.
            Shape: (batch_size, scene_size, obs_len + pred_len, 2)
        barycenters: Barycenters used for centering.
            Shape: (batch_size, 2)

    Returns:
        Un-centered scene tensors.
            Shape: (batch_size, scene_size, obs_len + pred_len, 2)
    """
    return scene_BSA2 + barycenters.unsqueeze(1).unsqueeze(1)


def _generate_random_scene_mask(num_scenes: int, prob: float, device: torch.device) -> Tensor:
    """Generate a random mask for applying transformations.

    Args:
        num_scenes: Number of scenes in the batch.
        prob: Probability of applying the transformation.
        device: Device to create the mask on.

    Returns:
        Random mask for applying the transformation.
            Shape: (batch_size,)
    """

    scene_probs_b = torch.rand(num_scenes, device=device)
    return scene_probs_b < prob
