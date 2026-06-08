import os
import subprocess
from typing import Literal, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import yaml

import rl_detect.utils as utils
import rl_detect.model.metrics as metrics


def gen_noise(shape: tuple[int, ...],
              noise_distrib: Literal['gaussian', 'uniform'],
              generator: torch.Generator | None = None,
              device=None):
    if noise_distrib == 'gaussian':
        return torch.randn(shape, generator=generator, device=device)
    elif noise_distrib == 'uniform':
        return torch.rand(shape, generator=generator, device=device)
    else:
        raise ValueError(f'Noise type {noise_distrib} not supported')


def handle_noise(batch_size: int,
                 num_samples: int,
                 scene_idx_B: torch.Tensor,
                 noise_dim: int,
                 noise_distrib: Literal['gaussian', 'uniform'],
                 noise_type: str,
                 noise: torch.Tensor | None,
                 device: torch.device) -> torch.Tensor:
    sampling_required = num_samples > 1 or noise is not None
    if sampling_required and noise_dim <= 0:
        raise ValueError('Cannot sample multiple trajectories '
                         'without noise')

    if noise is not None:
        # Check if noise is of correct shape.
        if noise_type == 'local':
            if noise.shape != (batch_size, num_samples, noise_dim):
                raise ValueError(
                    f'Noise shape must be '
                    f'({batch_size}, {num_samples}, {noise_dim})'
                )
            noise_BKL = noise

        else:
            # U: unique scenes (number of scenes).
            num_scenes = scene_idx_B.unique().size(0)
            if noise.shape != (num_scenes, num_samples, noise_dim):
                raise ValueError(
                    f'Noise shape must be ({num_samples}, {noise_dim})'
                )
            noise_UKL = noise
            noise_BKL = noise_UKL[scene_idx_B]

    else:
        # Generate noise.
        if noise_type == 'local':
            noise_BKL = gen_noise(
                (batch_size, num_samples, noise_dim),
                noise_distrib,
                device=device
            )
        else:
            # U: unique scenes (number of scenes).
            num_scenes = scene_idx_B.unique().size(0)
            noise_UKL = gen_noise(
                (num_scenes, num_samples, noise_dim),
                noise_distrib,
                device=device
            )
            noise_BKL = noise_UKL[scene_idx_B]

    return noise_BKL


@torch.no_grad()
def closest_sample_index(samples_BKC: Tensor,
                         target_BC: Tensor,
                         metric: Literal['ade', 'fde', 'goal']
                         ) -> Tensor:
    """Compute the index of the closest sample to the ground truth
    for each element in the batch.

    Args:
        samples_BKC: Samples. Shape: (batch_size, num_samples, *, C).
        target_BC: Target trajectories. Shape: (batch_size, *, C).
        metric: Metric to use for computing the distance.

    Returns:
        Index of the closest sample for each element in the batch.
        Shape: (batch_size,).
    """

    target_BKC = target_BC.unsqueeze(1).expand_as(samples_BKC)

    if metric == 'ade':
        # Compute ADE for each sample of each pedestrian.
        distances_BK = metrics.compute_ade(samples_BKC, target_BKC)
    elif metric == 'fde':
        # Compute FDE for each sample of each pedestrian.
        distances_BK = metrics.compute_fde(samples_BKC, target_BKC)
    elif metric == 'goal':
        # Compute goal distance for each sample of each pedestrian.
        distances_BK = torch.norm(samples_BKC - target_BKC, dim=-1)
    else:
        raise ValueError(f'Invalid metric: {metric}')

    # Compute the index of the closest sample for each pedestrian.
    closest_sample_index_B = distances_BK.argmin(dim=-1)

    return closest_sample_index_B


@torch.no_grad()
def check_env_collisions(traj_ZP2,
                         map_mask_B1HW,
                         scene_transform_matrix_B33,
                         homography_meters2mask_B33):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters. Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
        homography_meters2mask: Homography matrix to convert meters to mask coordinates.

    Returns:
        Boolean tensor indicating collisions. Shape: (batch_size,).
    """

    # Get dimensions.
    num_traj, _, _ = traj_ZP2.shape
    _, _, H, W = map_mask_B1HW.shape
    num_samples = num_traj // map_mask_B1HW.shape[0]

    # Convert map mask to boolean.
    map_mask_bool_B1HW = map_mask_B1HW > 0.5

    # Combine transformations into a single matrix multiplication.
    combined_transform_B33 = torch.bmm(
        scene_transform_matrix_B33,
        homography_meters2mask_B33
    )

    map_mask_bool_Z1HW = map_mask_bool_B1HW.repeat_interleave(num_samples, dim=0)
    combined_transform_Z33 = combined_transform_B33.repeat_interleave(num_samples, dim=0)

    # Transform all trajectories at once.
    # TODO: check if it works
    traj_BP2 = utils.project_batched(traj_ZP2, combined_transform_Z33)

    # Calculate bounds for all trajectories at once.
    x_coords = traj_BP2[..., 0]
    y_coords = traj_BP2[..., 1]

    # Check bounds once for all points.
    in_bounds = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)

    # Vectorized collision detection.
    collisions_B = torch.zeros(num_traj, dtype=torch.bool, device=traj_BP2.device)

    # TODO: maybe switch to other implementation
    for i in range(num_traj):
        if not in_bounds[i].any():
            continue

        # Extract relevant coordinates for this trajectory
        valid_y = y_coords[i][in_bounds[i]].long()
        valid_x = x_coords[i][in_bounds[i]].long()

        # Check if any point in trajectory overlaps with non-walkable area
        map_mask_bool_1HW = map_mask_bool_Z1HW[i]
        if not map_mask_bool_1HW[0, valid_y, valid_x].all():
            collisions_B[i] = True

    return collisions_B


@torch.no_grad()
def check_env_collisions_before_batch(traj_BP2,
                        map_mask_1HW,
                        scene_transform_matrix,
                        homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters. Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
        homography_meters2mask: Homography matrix to convert meters to mask coordinates.

    Returns:
        Boolean tensor indicating collisions. Shape: (batch_size,).
    """
    # Early return if no map mask
    if map_mask_1HW is None:
        return torch.zeros(traj_BP2.shape[0], dtype=torch.bool, device=traj_BP2.device)

    # Get dimensions
    num_traj, _, _ = traj_BP2.shape
    _, H, W = map_mask_1HW.shape

    # Convert map mask to boolean once
    map_mask_bool_1HW = map_mask_1HW > 0.5

    # Combine transformations into a single matrix multiplication
    combined_transform = torch.matmul(
        torch.inverse(scene_transform_matrix),
        homography_meters2mask
    )

    # Transform all trajectories at once
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP2 = torch.matmul(traj_BP3, combined_transform.T)[..., :2]

    # Calculate bounds for all trajectories at once
    x_coords = traj_BP2[..., 0]
    y_coords = traj_BP2[..., 1]

    # Check bounds once for all points
    in_bounds = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)

    # Vectorized collision detection
    collisions = torch.zeros(num_traj, dtype=torch.bool, device=traj_BP2.device)

    for i in range(num_traj):
        if not in_bounds[i].any():
            continue

        # Extract relevant coordinates for this trajectory
        valid_y = y_coords[i][in_bounds[i]].long()
        valid_x = x_coords[i][in_bounds[i]].long()

        # Check if any point in trajectory overlaps with non-walkable area
        if not map_mask_bool_1HW[0, valid_y, valid_x].all():
            collisions[i] = True

    return collisions

@torch.no_grad()
def check_env_collisions_v2(traj_BP2,
                         map_mask_1HW,
                         scene_transform_matrix,
                         homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters.
            Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            The code will use the inverse of this matrix to undo
            data augmentation, for aligning the scene with the map.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.

    Returns:
        Boolean tensor indicating if the trajectories go over the
        non-walkable area. Shape: (batch_size,).
    """

    # Number of trajectories (batch size).
    num_traj, num_pred, _ = traj_BP2.shape

    # Mask size.
    _, H, W = map_mask_1HW.shape

    if map_mask_1HW is None:
        return torch.zeros(num_traj,
                           dtype=torch.bool,
                           device=traj_BP2.device)

    # Inverse transformation matrix.
    inv_transform_matrix = torch.inverse(scene_transform_matrix)

    # Undo the transformation (data augmentation).
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP3 = torch.matmul(traj_BP3, inv_transform_matrix.T)
    traj_BP2 = traj_BP3[..., :2]

    # To map pixel coordinates.
    traj_BP2 = utils.project(traj_BP2, homography_meters2mask)

    # Map mask (true for walkable area).
    # Works with both [0, 1] and [0, 255] input masks.
    map_mask_bool_1HW = map_mask_1HW > 0.5

    # For each of the B trajectories, find the minimum and maximum
    # x and y coordinates.
    min_x_B = traj_BP2[:, :, 0].min(dim=1).values.long().clamp(min=0, max=W-1)
    max_x_B = traj_BP2[:, :, 0].max(dim=1).values.long().clamp(min=0, max=W-1)
    min_y_B = traj_BP2[:, :, 1].min(dim=1).values.long().clamp(min=0, max=H-1)
    max_y_B = traj_BP2[:, :, 1].max(dim=1).values.long().clamp(min=0, max=H-1)

    # Max height and width across all trajectories.
    max_height = (max_y_B - min_y_B).max() + 1
    max_width = (max_x_B - min_x_B).max() + 1

    # Prepare trajectory indexing.
    i_BP = torch.arange(num_traj, device=traj_BP2.device)[:, None].expand(-1, num_pred)
    y_BP = traj_BP2[:, :, 1]
    x_BP = traj_BP2[:, :, 0]

    # Extract the sub-maps for each trajectory.
    sub_maps = []
    for i in range(num_traj):
        x0 = min_x_B[i]
        y0 = min_y_B[i]
        x1 = x0 + max_width
        y1 = y0 + max_height

        right_pad = max(x1 - W, 0)
        bottom_pad = max(y1 - H, 0)

        sub_map_HW = map_mask_bool_1HW[:, y0:min(y1, H), x0:min(x1, W)]

        if right_pad > 0 or bottom_pad > 0:
            sub_map_HW = torch.nn.functional.pad(
                sub_map_HW,
                (0, right_pad, 0, bottom_pad),
                value=True
            )

        sub_maps.append(sub_map_HW)

    # Stack the sub-maps into a tensor.
    sub_maps_BHW = torch.cat(sub_maps, dim=0)

    # Build trajectory mask.
    traj_mask_BHW = torch.zeros_like(sub_maps_BHW, dtype=torch.bool)

    # Mask for points that are in/out of image bounds.
    in_bounds_BP = (y_BP >= 0) & (y_BP < H) & (x_BP >= 0) & (x_BP < W)

    # Translate the coordinates to the sub-maps.
    y_BP -= min_y_B[:, None]
    x_BP -= min_x_B[:, None]

    y_BP = y_BP.long()
    x_BP = x_BP.long()

    # Filter out-of-bounds points.
    i_N = i_BP[in_bounds_BP]
    y_N = y_BP[in_bounds_BP]
    x_N = x_BP[in_bounds_BP]

    traj_mask_BHW[i_N, y_N, x_N] = True

    # Check if the predicted trajectory goes over the non-walkable area.
    overlap_BHW = traj_mask_BHW & (~sub_maps_BHW)
    overlap_B = overlap_BHW.any(dim=(1, 2))

    return overlap_B


@torch.no_grad()
def check_env_collisions_v3(traj_BP2,
                         map_mask_1HW,
                         scene_transform_matrix,
                         homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters.
            Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            The code will use the inverse of this matrix to undo
            data augmentation, for aligning the scene with the map.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.

    Returns:
        Boolean tensor indicating if the trajectories go over the
        non-walkable area. Shape: (batch_size,).
    """

    # Number of trajectories (batch size).
    num_traj, num_pred, _ = traj_BP2.shape

    # Mask size.
    _, H, W = map_mask_1HW.shape

    if map_mask_1HW is None:
        return torch.zeros(num_traj,
                           dtype=torch.bool,
                           device=traj_BP2.device)

    # Inverse transformation matrix.
    inv_transform_matrix = torch.inverse(scene_transform_matrix)

    # Undo the transformation (data augmentation).
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP3 = torch.matmul(traj_BP3, inv_transform_matrix.T)
    traj_BP2 = traj_BP3[..., :2]

    # To map pixel coordinates.
    traj_BP2 = utils.project(traj_BP2, homography_meters2mask)

    # Map mask.
    map_mask_bool_1HW = map_mask_1HW > 0.5

    # For each of the B trajectories, find the minimum and maximum
    # x and y coordinates.
    min_x_B = traj_BP2[:, :, 0].min(dim=1).values.long().clamp(max=W-1)
    max_x_B = traj_BP2[:, :, 0].max(dim=1).values.long().clamp(max=W-1)
    min_y_B = traj_BP2[:, :, 1].min(dim=1).values.long().clamp(max=H-1)
    max_y_B = traj_BP2[:, :, 1].max(dim=1).values.long().clamp(max=H-1)

    # Max height and width across all trajectories.
    max_height = (max_y_B - min_y_B).max() + 1
    max_width = (max_x_B - min_x_B).max() + 1

    # Extract the sub-maps for each trajectory.
    sub_maps = []
    for i in range(num_traj):
        x0 = min_x_B[i]
        y0 = min_y_B[i]
        x1 = x0 + max_width
        y1 = y0 + max_height

        right_pad = max(x1 - W, 0)
        bottom_pad = max(y1 - H, 0)

        sub_map_HW = map_mask_bool_1HW[:, y0:y1, x0:x1]

        if right_pad > 0 or bottom_pad > 0:
            sub_map_HW = torch.nn.functional.pad(
                sub_map_HW,
                (0, right_pad, 0, bottom_pad),
                value=True
            )

        sub_maps.append(sub_map_HW)

    # Stack the sub-maps.
    try:
        sub_maps_BHW = torch.cat(sub_maps, dim=0)
    except Exception as e:
        print(H, W)
        print(max_height, max_width)
        x1 = min_x_B + max_width
        y1 = min_y_B + max_height
        print(y1 > H, x1 > W)
        raise

    # Build trajectory mask.
    traj_mask_BHW = torch.zeros_like(sub_maps_BHW, dtype=torch.bool)


    ##############################
    # Superimpose the trajectories on the map.
    # for i in range(num_traj):
    #     traj_mask_SHW[i,
    #                   scene_pred_hat_SP2[i, :, 1].long(),
    #                   scene_pred_hat_SP2[i, :, 0].long()] = True
    # i_B1 = torch.arange(num_traj)[:, None]
    i_BP = torch.arange(num_traj, device=traj_BP2.device)[:, None].expand(-1, num_pred)
    y_BP = traj_BP2[:, :, 1]
    x_BP = traj_BP2[:, :, 0]

    # traj_mask_BHW[i_B1, y_BP, x_BP] = True
    #############################


    # Mask for points that are in/out of image bounds.
    in_bounds_BP = (y_BP >= 0) & (y_BP < H) & (x_BP >= 0) & (x_BP < W)

    # Translate the coordinates to the sub-maps.
    y_BP -= min_y_B[:, None]
    x_BP -= min_x_B[:, None]

    y_BP = y_BP.long()
    x_BP = x_BP.long()

    # Filter out-of-bounds points (and flatten indexing tensors).
    # N: total number of in-bounds points for all the B trajectories.
    # N = in_bounds_BP.sum()
    i_N = i_BP[in_bounds_BP]
    y_N = y_BP[in_bounds_BP]
    x_N = x_BP[in_bounds_BP]

    traj_mask_BHW[i_N, y_N, x_N] = True

    #############################

    # Check if the predicted trajectory goes over the non-walkable area.
    overlap_BHW = traj_mask_BHW & (~sub_maps_BHW)
    overlap_B = overlap_BHW.any(dim=(1, 2))

    return overlap_B

@torch.no_grad()
def check_env_collisions_old(traj_BP2,
                         map_mask_1HW,
                         scene_transform_matrix,
                         homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters.
            Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            The code will use the inverse of this matrix to undo
            data augmentation, for aligning the scene with the map.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.

    Returns:
        Boolean tensor indicating if the trajectories go over the
        non-walkable area. Shape: (batch_size,).
    """

    # Number of trajectories (batch size).
    num_traj, num_pred, _ = traj_BP2.shape

    if map_mask_1HW is None:
        return torch.zeros(num_traj,
                           dtype=torch.bool,
                           device=traj_BP2.device)

    # Inverse transformation matrix.
    inv_transform_matrix = torch.inverse(scene_transform_matrix)

    # Undo the transformation (data augmentation).
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP3 = torch.matmul(traj_BP3, inv_transform_matrix.T)
    traj_BP2 = traj_BP3[..., :2]

    # To map pixel coordinates.
    traj_BP2 = utils.project(traj_BP2, homography_meters2mask)

    # Map mask.
    map_mask_bool_1HW = map_mask_1HW > 0.5
    map_mask_bool_BHW = map_mask_bool_1HW.expand(num_traj, -1, -1)

    # Build trajectory mask.
    traj_mask_BHW = torch.zeros_like(map_mask_bool_BHW, dtype=torch.bool)


    ##############################
    # Superimpose the trajectories on the map.
    # for i in range(num_traj):
    #     traj_mask_SHW[i,
    #                   scene_pred_hat_SP2[i, :, 1].long(),
    #                   scene_pred_hat_SP2[i, :, 0].long()] = True
    # i_B1 = torch.arange(num_traj)[:, None]
    i_BP = torch.arange(num_traj, device=traj_BP2.device)[:, None].expand(-1, num_pred)
    y_BP = traj_BP2[:, :, 1].long()
    x_BP = traj_BP2[:, :, 0].long()
    # traj_mask_BHW[i_B1, y_BP, x_BP] = True
    #############################


    # Mask for points that are in/out of image bounds.
    _, H, W = map_mask_bool_BHW.shape
    in_bounds_BP = (y_BP >= 0) & (y_BP < H) & (x_BP >= 0) & (x_BP < W)

    # Filter out-of-bounds points (and flatten indexing tensors).
    # N: total number of in-bounds points for all the B trajectories.
    # N = in_bounds_BP.sum()
    i_N = i_BP[in_bounds_BP]
    y_N = y_BP[in_bounds_BP]
    x_N = x_BP[in_bounds_BP]

    traj_mask_BHW[i_N, y_N, x_N] = True

    #############################

    # Check if the predicted trajectory goes over the non-walkable area.
    overlap_BHW = traj_mask_BHW & (~map_mask_bool_BHW)
    overlap_B = overlap_BHW.any(dim=(1, 2))

    return overlap_B


def augment_traj_resolution(traj_ST2: torch.Tensor, parts: int) -> torch.Tensor:
    """Augment trajectory resolution by adding `parts` interpolated points
    between each pair of consecutive points."""

    # S: scene size (number of trajectories)
    # T: trajectory length
    # P: number of parts (or number of parts - 1)
    # 2: x, y coordinates

    S, _, _ = traj_ST2.shape

    # Create interpolation coefficients
    coeffs_P = torch.linspace(0, 1, parts + 2, device=traj_ST2.device)
    coeffs_P = coeffs_P[:-1]
    coeffs_P1 = coeffs_P[:, None]

    # Start and end points for the interpolation, with an extra dimension
    # for the parts.
    start_points_ST12 = traj_ST2[:, :-1, None, :]
    end_points_ST12 = traj_ST2[:, 1:, None, :]

    # Interpolate between the start and end points.
    interpolated_points_STL2 = \
        start_points_ST12 + coeffs_P1 * (end_points_ST12 - start_points_ST12)

    # Concatenate the interpolated points along the parts dimension
    # into the time dimension.
    interpolated_points_ST2 = interpolated_points_STL2.reshape(S, -1, 2)

    # Add the last point of the original trajectory.
    interpolated_points_ST2 = \
        torch.cat((interpolated_points_ST2, traj_ST2[:, -1:, :]), dim=1)

    return interpolated_points_ST2


@torch.no_grad()
def extract_patches_batched_cpu(traj_BO2: torch.Tensor,
                                map_mask_B1HW: torch.Tensor,
                                scene_transform_matrix_B33: torch.Tensor,
                                homography_meters2mask_B33: torch.Tensor,
                                patch_size_px: int,
                                back_dist_px: int) -> torch.Tensor:
    """Extract patches from the map mask for each person in the scene.

    Args:
        scene_SO2: Scene tensor. Expected in absolute coordinates and
            in meters. Need at least 2 timesteps.
            Shape: (scene_size, obs_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            To undo data augmentation, for aligning the scene with the map.
        dataset_name: Name of the dataset.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.
        patch_size_px: Patch size in pixels (square patches).
        back_dist_px: Distance to the back of the person in pixels.

    Returns:
        Mask patches tensor. Shape: (scene_size, 1, patch_size, patch_size).
    """

    # TODO: parametrize
    # PATCH_SIZE_M = 10      # meters
    # PATCH_SIZE_PX = 100    # pixels
    # BACK_DIST_M = 1        # meters
    # BACK_DIST_PX = 10      # pixels

    # Last 2 positions.
    curr_pos_B2 = traj_BO2[:, -1, :]
    prev_pos_B2 = traj_BO2[:, -2, :]

    # # To homogeneous coordinates.
    # ones = torch.ones(curr_pos_B2.shape[0], 1, device=curr_pos_B2.device)
    # curr_pos_B3 = torch.cat((curr_pos_B2, ones), dim=1)
    # prev_pos_B3 = torch.cat((prev_pos_B2, ones), dim=1)

    # # Apply inverse transform (undo data augmentation).
    # curr_pos_B3 = torch.matmul(scene_transform_matrix_B33, curr_pos_B3.T).T
    # prev_pos_B3 = torch.matmul(scene_transform_matrix_B33, prev_pos_B3.T).T

    # # Back to world coordinates.
    # curr_pos_B2 = curr_pos_B3[:, :2]
    # prev_pos_B2 = prev_pos_B3[:, :2]

    # Combine transformations into a single matrix multiplication.
    combined_transform_B33 = torch.bmm(
        scene_transform_matrix_B33,
        homography_meters2mask_B33
    )

    # Convert world coordinates to mask pixel coordinates.
    # TODO: try to fuse in a single operation for speedup.
    curr_pos_B2 = utils.project_batched(curr_pos_B2, combined_transform_B33)
    prev_pos_B2 = utils.project_batched(prev_pos_B2, combined_transform_B33)

    # equals_B = np.isclose(curr_pos_B2, prev_pos_B2).all(axis=1)
    equals_B = (curr_pos_B2 == prev_pos_B2).all(axis=1)

    if equals_B.any():
        # If positions are the same, randomly perturb the prev.
        prev_pos_B2 = prev_pos_B2.clone()

        # Sample angles uniformly between 0 and 2pi for the perturbation.
        angles_B = 2 * torch.pi * torch.rand(equals_B.sum(),
                                             device=traj_BO2.device)
        delta_B2 = torch.stack([torch.cos(angles_B),
                                torch.sin(angles_B)], dim=-1)

        # Apply perturbations to the previous positions.
        prev_pos_B2[equals_B] += delta_B2

    # Compute forward direction.
    fwd_dir_B2 = curr_pos_B2 - prev_pos_B2
    fwd_dir_B2 /= torch.norm(fwd_dir_B2, dim=1, keepdim=True)

    # Compute left direction.
    left_dir_B2 = torch.stack((-fwd_dir_B2[:, 1], fwd_dir_B2[:, 0]), dim=1)

    # Compute back position.
    back_pos_B2 = curr_pos_B2 - back_dist_px * fwd_dir_B2

    # Compute corners.
    back_left_B2 = back_pos_B2 + left_dir_B2 * patch_size_px / 2
    back_right_B2 = back_pos_B2 - left_dir_B2 * patch_size_px / 2
    front_left_B2 = back_left_B2 + fwd_dir_B2 * patch_size_px
    front_right_B2 = back_right_B2 + fwd_dir_B2 * patch_size_px

    # Convert to numpy since opencv requires it.
    # TODO: check if i can implement getAffineTransform in pytorch.
    corners_4B2 = np.array([back_left_B2.cpu(),
                            back_right_B2.cpu(),
                            front_left_B2.cpu(),
                            front_right_B2.cpu()],
                            dtype=np.float32)
    target_corners_42 = np.array([[0, 0],
                                    [patch_size_px, 0],
                                    [0, patch_size_px],
                                    [patch_size_px, patch_size_px]],
                                    dtype=np.float32)


    mask_patches = []
    for i in range(traj_BO2.shape[0]):
        # cv.warpAffine requires (H, W) format.
        map_mask_HW = map_mask_B1HW[i, 0].cpu().numpy()

        # Extract patches.
        # Affine transform computation requires 3 points.
        patch_affine_transform = cv.getAffineTransform(
            np.ascontiguousarray(corners_4B2[:3, i]),
            target_corners_42[:3]
        )

        mask_patch = cv.warpAffine(map_mask_HW,
                                   patch_affine_transform,
                                   (patch_size_px, patch_size_px),
                                   borderValue=255)
        mask_patch = mask_patch / 255.0
        mask_patches.append(mask_patch)

    mask_patches = np.array(mask_patches)

    mask_tensor_B1HW = \
        torch.from_numpy(mask_patches).\
                unsqueeze(1).\
                float().\
                to(traj_BO2.device)

    return mask_tensor_B1HW


################################################################################
################################################################################
################################################################################

# TODO: this implementation fill with 0s the padding
@torch.no_grad()
def extract_patches_batched(traj_BO2: torch.Tensor,
                            map_mask_B1HW: torch.Tensor,
                            scene_transform_matrix_B33: torch.Tensor,
                            homography_meters2mask_B33: torch.Tensor,
                            patch_size_px: int,
                            back_dist_px: int) -> torch.Tensor:
    """
    Batched patch extraction using torch functions (replacing OpenCV).
    See original function for argument descriptions.
    Assumes map_mask_B1HW is a tensor of shape (B,1,H,W) with mask values in [0,255].
    """
    device = traj_BO2.device
    B = traj_BO2.shape[0]

    # 1. Compute current and previous positions in world coordinates.
    curr_pos_B2 = traj_BO2[:, -1, :]  # (B,2)
    prev_pos_B2 = traj_BO2[:, -2, :]  # (B,2)

    # (The code block that undoes data augmentation was commented out.)
    # Instead, we combine the scene transform and the meters-to-mask homography.
    combined_transform_B33 = torch.bmm(
        homography_meters2mask_B33, scene_transform_matrix_B33
    )

    # Project world coordinates into mask pixel coordinates.
    # (Assumes that utils.project_batched applies a homogeneous transform.)
    curr_pos_B2 = utils.project_batched(curr_pos_B2, combined_transform_B33)
    prev_pos_B2 = utils.project_batched(prev_pos_B2, combined_transform_B33)

    # If current and previous positions coincide, perturb the previous position.
    equals_B = (curr_pos_B2 == prev_pos_B2).all(dim=1)
    if equals_B.any():
        prev_pos_B2 = prev_pos_B2.clone()
        angles = 2 * torch.pi * torch.rand(equals_B.sum(), device=device)
        delta = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        prev_pos_B2[equals_B] += delta

    # 2. Compute the (unit) forward and left directions.
    fwd_dir_B2 = curr_pos_B2 - prev_pos_B2
    fwd_dir_B2 = fwd_dir_B2 / torch.norm(fwd_dir_B2, dim=1, keepdim=True)
    left_dir_B2 = torch.stack([-fwd_dir_B2[:, 1], fwd_dir_B2[:, 0]], dim=1)

    # 3. Compute the “back” position and the three corners we need.
    #    We define the patch such that its “back” is at a fixed offset behind the person.
    back_pos_B2 = curr_pos_B2 - back_dist_px * fwd_dir_B2  # (B,2)
    # The three corners (in the map mask image) that we will use to define an affine transform:
    #    - back_left: the back-center shifted left half the patch width.
    #    - back_right: the back-center shifted right half the patch width.
    #    - front_left: the front-left (obtained by moving forward from back_left by patch_size_px).
    back_left_B2  = back_pos_B2 + left_dir_B2 * (patch_size_px / 2)
    back_right_B2 = back_pos_B2 - left_dir_B2 * (patch_size_px / 2)
    front_left_B2 = back_left_B2 + fwd_dir_B2 * patch_size_px
    # (We could also compute front_right, but three points suffice.)

    # 4. For each patch, compute the affine transform that maps patch pixel coordinates (in the target)
    #    to the corresponding pixel coordinates in the map.
    #
    # In OpenCV, one would use:
    #   patch_affine_transform = cv.getAffineTransform(src_points, dst_points)
    # where src_points are the 3 computed corners and dst_points are the 3 corners of a square:
    #   target_corners = [[0, 0], [patch_size_px, 0], [0, patch_size_px]].
    #
    # Here we compute it directly.
    # For each patch, let:
    #   src0 = back_left, src1 = back_right, src2 = front_left.
    # We want M (2x3) such that:
    #   M @ [0, 0, 1]^T = src0,
    #   M @ [patch_size_px, 0, 1]^T = src1,
    #   M @ [0, patch_size_px, 1]^T = src2.
    #
    # This gives:
    #   M[:,2] = src0,
    #   M[:,0] = (src1 - src0) / patch_size_px,
    #   M[:,1] = (src2 - src0) / patch_size_px.
    #
    M = torch.empty(B, 2, 3, device=device)
    M[:, :, 2] = back_left_B2  # broadcast: M[i, :, 2] = back_left of patch i.
    M[:, :, 0] = (back_right_B2 - back_left_B2) / patch_size_px
    M[:, :, 1] = (front_left_B2 - back_left_B2) / patch_size_px

    # 5. Convert this transform into the “normalized” coordinates required by grid_sample.
    #
    # grid_sample expects an affine matrix theta of shape (B,2,3) such that for an output grid (in normalized
    # coordinates, i.e. in [-1,1]), the source sampling point is computed as:
    #    [x_s, y_s] = theta @ [x_t, y_t, 1]
    #
    # Our computed matrix M maps patch pixel coordinates (in [0, patch_size_px]) to map pixel coordinates.
    # To use grid_sample we must:
    #
    #   1. Convert patch pixel coordinates (target) to normalized coordinates.
    #   2. Convert map pixel coordinates (source) to normalized coordinates.
    #
    # That is, we need to incorporate two normalization transforms.
    #
    # Define T_patch_inv: 3x3 transform converting patch normalized coordinates (in [-1,1]) into patch pixels.
    # With the convention that:
    #   x_pixel = ( (x_norm + 1) * (patch_size_px - 1) / 2 )
    #
    # Similarly, define T_in: 3x3 transform converting map pixel coordinates into normalized coordinates.
    # For an input map of size (H, W) (width = W, height = H):
    #   x_norm = 2 * x_pixel/(W - 1) - 1
    #   y_norm = 2 * y_pixel/(H - 1) - 1
    #
    # Thus we set:
    H = map_mask_B1HW.shape[2]
    W = map_mask_B1HW.shape[3]

    # T_in: from input pixels to normalized coordinates.
    T_in = torch.tensor([[2/(W-1), 0, -1],
                           [0, 2/(H-1), -1],
                           [0, 0, 1]], device=device, dtype=torch.float32)
    # T_patch_inv: from normalized patch coordinates to patch pixels.
    T_patch_inv = torch.tensor([[(patch_size_px - 1) / 2, 0, (patch_size_px - 1) / 2],
                                 [0, (patch_size_px - 1) / 2, (patch_size_px - 1) / 2],
                                 [0, 0, 1]], device=device, dtype=torch.float32)
    # Expand these to batch (they are the same for all patches).
    T_in = T_in.unsqueeze(0).expand(B, 3, 3)       # (B,3,3)
    T_patch_inv = T_patch_inv.unsqueeze(0).expand(B, 3, 3)  # (B,3,3)

    # Augment M (which is 2x3) to a 3x3 by adding [0, 0, 1] as the last row.
    last_row = torch.tensor([0, 0, 1], device=device, dtype=torch.float32).view(1, 1, 3).expand(B, 1, 3)
    M_aug = torch.cat([M, last_row], dim=1)  # (B,3,3)

    # Our overall mapping from patch normalized coordinates to input normalized coordinates is:
    #   theta_full = T_in @ M_aug @ T_patch_inv
    theta_full = torch.bmm(torch.bmm(T_in, M_aug), T_patch_inv)  # (B, 3, 3)
    # grid_sample only requires the 2x3 part.
    theta = theta_full[:, :2, :]  # (B,2,3)

    # 6. Create the grid and sample.
    # Normalize: grid_sample expects the output grid in normalized coordinates.
    grid = F.affine_grid(theta, size=(B, 1, patch_size_px, patch_size_px),
                         align_corners=True)

    # Note: The OpenCV version uses borderValue=255, then divides the result by 255.
    # Here we assume that map_mask_B1HW has values in [0, 255]. We first convert to float and scale.
    map_mask_norm = map_mask_B1HW.float() / 255.0

    # Use grid_sample. We use mode='bilinear' (as in warpAffine) and padding_mode='border'
    # so that out-of-bound locations are filled with the border value (which, if the border is 255,
    # becomes 1 after normalization).
    mask_patches = F.grid_sample(map_mask_norm, grid, mode='bilinear',
                                 padding_mode='border', align_corners=True)
    # mask_patches is (B, 1, patch_size_px, patch_size_px)

    return mask_patches


################################################################################
################################################################################
################################################################################


@torch.no_grad()
def extract_patches(scene_SO2: torch.Tensor,
                    map_mask_1HW: torch.Tensor,
                    scene_transform_matrix: torch.Tensor,
                    homography_meters2mask: torch.Tensor,
                    patch_size_px: int,
                    back_dist_px: int) -> torch.Tensor:
    """Extract patches from the map mask for each person in the scene.

    Args:
        scene_SO2: Scene tensor. Expected in absolute coordinates and
            in meters. Need at least 2 timesteps.
            Shape: (scene_size, obs_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
            Expected to be uint8 with values in [0, 255].
        scene_transform_matrix: Scene transform matrix.
            To undo data augmentation, for aligning the scene with the map.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.
        patch_size_px: Patch size in pixels (square patches).
        back_dist_px: Distance to the back of the person in pixels.

    Returns:
        Mask patches tensor. Shape: (scene_size, 1, patch_size, patch_size).
        The patches are float [0, 1].
    """
    """Extract patches from the map mask for each person in the scene.

    Args:
        scene_SO2: Scene tensor. Expected in absolute coordinates and
            in meters. Need at least 2 timesteps.
            Shape: (scene_size, obs_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            To undo data augmentation, for aligning the scene with the map.
        dataset_name: Name of the dataset.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.
        patch_size_px: Patch size in pixels (square patches).
        back_dist_px: Distance to the back of the person in pixels.

    Returns:
        Mask patches tensor. Shape: (scene_size, 1, patch_size, patch_size).
    """

    # TODO: parametrize
    # PATCH_SIZE_M = 10      # meters
    # PATCH_SIZE_PX = 100    # pixels
    # BACK_DIST_M = 1        # meters
    # BACK_DIST_PX = 10      # pixels

    # Last 2 positions.
    curr_pos_S2 = scene_SO2[:, -1, :]
    prev_pos_S2 = scene_SO2[:, -2, :]

    # To homogeneous coordinates.
    ones = torch.ones(curr_pos_S2.shape[0], 1, device=curr_pos_S2.device)
    curr_pos_S3 = torch.cat((curr_pos_S2, ones), dim=1)
    prev_pos_S3 = torch.cat((prev_pos_S2, ones), dim=1)

    # Apply inverse transform (undo data augmentation).
    curr_pos_S3 = torch.matmul(scene_transform_matrix, curr_pos_S3.T).T
    prev_pos_S3 = torch.matmul(scene_transform_matrix, prev_pos_S3.T).T

    # Back to world coordinates.
    curr_pos_S2 = curr_pos_S3[:, :2]
    prev_pos_S2 = prev_pos_S3[:, :2]

    # Convert world coordinates to mask pixel coordinates.
    # TODO: try to fuse in a single operation for speedup.
    curr_pos_S2 = utils.project(curr_pos_S2, homography_meters2mask)
    prev_pos_S2 = utils.project(prev_pos_S2, homography_meters2mask)

    # equals_S = np.isclose(curr_pos_S2, prev_pos_S2).all(axis=1)
    equals_S = (curr_pos_S2 == prev_pos_S2).all(axis=1)

    if equals_S.any():
        # If positions are the same, randomly perturb the prev.
        prev_pos_S2 = prev_pos_S2.clone()

        # Sample angles uniformly between 0 and 2pi for the perturbation.
        angles_S = 2 * torch.pi * torch.rand(equals_S.sum(),
                                             device=scene_SO2.device)
        delta_S2 = torch.stack([torch.cos(angles_S),
                                torch.sin(angles_S)], dim=-1)

        # Apply perturbations to the previous positions.
        prev_pos_S2[equals_S] += delta_S2

    # Compute forward direction.
    fwd_dir_S2 = curr_pos_S2 - prev_pos_S2
    fwd_dir_S2 /= torch.norm(fwd_dir_S2, dim=1, keepdim=True)

    # Compute left direction.
    left_dir_S2 = torch.stack((-fwd_dir_S2[:, 1], fwd_dir_S2[:, 0]), dim=1)

    # Compute back position.
    back_pos_S2 = curr_pos_S2 - back_dist_px * fwd_dir_S2

    # Compute corners.
    back_left_S2 = back_pos_S2 + left_dir_S2 * patch_size_px / 2
    back_right_S2 = back_pos_S2 - left_dir_S2 * patch_size_px / 2
    front_left_S2 = back_left_S2 + fwd_dir_S2 * patch_size_px
    front_right_S2 = back_right_S2 + fwd_dir_S2 * patch_size_px

    # Convert to numpy since opencv requires it.
    # TODO: check if i can implement getAffineTransform in pytorch.
    corners_4S2 = np.array([back_left_S2.cpu(),
                            back_right_S2.cpu(),
                            front_left_S2.cpu(),
                            front_right_S2.cpu()],
                            dtype=np.float32)
    target_corners_42 = np.array([[0, 0],
                                    [patch_size_px, 0],
                                    [0, patch_size_px],
                                    [patch_size_px, patch_size_px]],
                                    dtype=np.float32)

    # cv.warpAffine requires (H, W) format.
    map_mask_HW = (map_mask_1HW.squeeze(0).cpu().numpy()).astype(np.uint8)

    mask_patches = []
    for i in range(scene_SO2.shape[0]):
        # Extract patches.
        # Affine transform computation requires 3 points.
        patch_affine_transform = cv.getAffineTransform(
            np.ascontiguousarray(corners_4S2[:3, i]),
            target_corners_42[:3]
        )

        mask_patch = cv.warpAffine(map_mask_HW,
                                    patch_affine_transform,
                                    (patch_size_px, patch_size_px),
                                    borderValue=255)
        mask_patch = mask_patch / 255.0
        mask_patches.append(mask_patch)

    mask_patches = np.array(mask_patches)

    mask_tensor_S1HW = \
        torch.from_numpy(mask_patches).\
                unsqueeze(1).\
                float().\
                to(scene_SO2.device)

    return mask_tensor_S1HW


def lstm_weights_init(lstm_module):
    hidden_size = lstm_module.hidden_size

    for name, param in lstm_module.named_parameters():
        if 'weight_ih' in name:
            # Initialize input weights with xaiver uniform for each gate.
            # (W_ii|W_if|W_ig|W_io)
            W_ii = param[:hidden_size]
            W_if = param[hidden_size:2*hidden_size]
            W_ig = param[2*hidden_size:3*hidden_size]
            W_io = param[3*hidden_size:]

            torch.nn.init.xavier_uniform_(W_ii)
            torch.nn.init.xavier_uniform_(W_if)
            torch.nn.init.xavier_uniform_(W_ig)
            torch.nn.init.xavier_uniform_(W_io)

        elif 'weight_hh' in name:
            # Initialize hidden weights with orthogonal for each gate.
            # (W_hi|W_hf|W_hg|W_ho)
            W_hi = param[:hidden_size]
            W_hf = param[hidden_size:2*hidden_size]
            W_hg = param[2*hidden_size:3*hidden_size]
            W_ho = param[3*hidden_size:]

            torch.nn.init.orthogonal_(W_hi)
            torch.nn.init.orthogonal_(W_hf)
            torch.nn.init.orthogonal_(W_hg)
            torch.nn.init.orthogonal_(W_ho)

        elif 'bias' in name:
            # Initialize input and hidden biases with zeros, except the forget
            # gate bias, which is initialized to 1.
            # (b_i|b_f|b_g|b_o)
            b_i = param[:hidden_size]
            b_f = param[hidden_size:2*hidden_size]
            b_g = param[2*hidden_size:3*hidden_size]
            b_o = param[3*hidden_size:]

            torch.nn.init.zeros_(b_i)
            torch.nn.init.ones_(b_f)
            torch.nn.init.zeros_(b_g)
            torch.nn.init.zeros_(b_o)


def gru_weights_init(gru_module):
    hidden_size = gru_module.hidden_size

    for name, param in gru_module.named_parameters():
        if 'weight_ih' in name:
            # Initialize input weights with xavier uniform for each gate
            # (W_ir|W_iz|W_in)
            W_ir = param[:hidden_size]  # reset gate
            W_iz = param[hidden_size:2*hidden_size]  # update gate
            W_in = param[2*hidden_size:]  # new gate

            torch.nn.init.xavier_uniform_(W_ir)
            torch.nn.init.xavier_uniform_(W_iz)
            torch.nn.init.xavier_uniform_(W_in)

        elif 'weight_hh' in name:
            # Initialize hidden weights with orthogonal for each gate
            # (W_hr|W_hz|W_hn)
            W_hr = param[:hidden_size]  # reset gate
            W_hz = param[hidden_size:2*hidden_size]  # update gate
            W_hn = param[2*hidden_size:]  # new gate

            torch.nn.init.orthogonal_(W_hr)
            torch.nn.init.orthogonal_(W_hz)
            torch.nn.init.orthogonal_(W_hn)

        elif 'bias' in name:
            # Initialize biases with zeros, except the update gate bias
            # which is initialized to 1 (similar to LSTM's forget gate)
            # (b_ir|b_iz|b_in) or (b_hr|b_hz|b_hn)
            b_r = param[:hidden_size]  # reset gate
            b_z = param[hidden_size:2*hidden_size]  # update gate
            b_n = param[2*hidden_size:]  # new gate

            torch.nn.init.zeros_(b_r)
            torch.nn.init.ones_(b_z)  # bias towards remembering
            torch.nn.init.zeros_(b_n)


def init_weights(arch: Literal['lstm', 'gru'], module):
    if arch == 'lstm':
        lstm_weights_init(module)
    elif arch == 'gru':
        gru_weights_init(module)
    else:
        raise ValueError(f'Architecture {arch} not supported')


def git_info():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass

    return sha, diff, branch


def save_results(filename: str,
                 results: dict[str, float] | list[dict[str, float]]):
    # Add git info.
    sha, diff, branch = git_info()
    git_obj = {
        "sha": sha,
        "diff": diff,
        "branch": branch,
    }

    # Build output object.
    if isinstance(results, list):
        output = {
            "git": git_obj,
            "results": results,
        }
    else:
        output = {
            "git": git_obj,
            **results,
        }

    # Store output object as YAML.
    with open(filename, 'w') as file:
        yaml.dump(output, file)


def save_predictions(folder: str, out_dict: dict[str, torch.Tensor]):
    os.makedirs(folder, exist_ok=True)

    for key, value in out_dict.items():
        torch.save(value, os.path.join(folder, f'{key}.pt'))
