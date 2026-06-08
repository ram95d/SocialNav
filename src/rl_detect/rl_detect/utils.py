"""Utility functions."""

import json

import torch
import numpy as np


def default_camera_params(img_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Returns the default camera parameters.

    A reasonable guess.

    Args:
        img_shape: The shape of the image.

    Returns:
        The tuple (camera_matrix, distortion_coefficients), where
        camera_matrix is the matrix of the camera intrinsic parameters,
        and distortion_coefficients are the distortion coefficients
        of the camera.
    """

    cam_mat = np.array([[img_shape[0], 0, img_shape[1]/2],
                        [0, img_shape[0], img_shape[0]/2],
                        [0, 0, 1]])

    dist_coeffs = np.zeros((1, 5))

    return cam_mat, dist_coeffs


def load_camera_params(camera_params_path: str | None) -> tuple[np.ndarray,
                                                                np.ndarray]:
    """Retrieves the camera parameters (json file).

    Args:
        camera_params_path: Path to the camera parameters JSON file.

    Returns:
        The tuple (camera_matrix, distortion_coefficients), where
        camera_matrix is the matrix of the camera intrinsic parameters,
        and distortion_coefficients are the distortion coefficients
        of the camera.
    """

    # Load the camera parameters from the JSON file.
    with open(camera_params_path, 'r') as f:
        params = json.load(f)

    # Check presence of expected fields.
    if 'camera_matrix' not in params \
        or 'distortion_coefs' not in params:
        raise RuntimeError('Invalid JSON format.')

    cam_mtx = np.array(params['camera_matrix'])
    dist_coeffs = np.array(params['distortion_coefs'])

    return cam_mtx, dist_coeffs


def save_camera_params(cam_mtx: np.ndarray,
                       dist_coeffs: np.ndarray,
                       camera_params_path: str) -> None:
     """Saves the camera parameters (json file).

     Args:
          cam_mtx: The matrix of the camera intrinsic parameters.
          dist_coeffs: The distortion coefficients of the camera.
          camera_params_path: Path to the camera parameters JSON file.
     """

     # Save the camera parameters to the JSON file.
     with open(camera_params_path, 'w') as f:
        json.dump({
            'camera_matrix': cam_mtx.tolist(),
            'distortion_coefs': dist_coeffs.tolist()
        }, f, indent=4)


def load_homography(homography_path: str) -> np.ndarray:
    """Retrieves the homography matrix (json file).

    Args:
        homography_path: Path to the homography JSON file.

    Returns:
        The homography matrix.
    """

    # Load the homography from the JSON file.
    with open(homography_path, 'r') as f:
        H = np.array(json.load(f))

    return H


def save_homography(H: np.ndarray, homography_path: str) -> None:
    """Saves the homography matrix (json file).

    Args:
        H: The homography matrix.
        homography_path: Path to the homography JSON file.
    """

    # Save the homography to the JSON file.
    with open(homography_path, 'w') as f:
        json.dump(H.tolist(), f, indent=4)


def project(xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Projects points using the given homography matrix.

    Args:
        xy: Points to be projected.
            Shape: (*, 2).
        homography: Homography matrix.
            Shape: (3, 3).

    Returns:
        Projected points.
        Shape: (*, 2).
    """

    # Original shape.
    shape = list(xy.shape)
    shape[-1] = 1

    # Homogeneous coordinates.
    xy_hom = torch.cat((xy, torch.ones(shape, device=xy.device)), dim=-1)

    # Project the points.
    xy_proj = xy_hom @ homography.T

    # Euclidean coordinates.
    xy_proj = xy_proj / xy_proj[..., 2:]
    xy_proj = xy_proj[..., :2]

    return xy_proj


def project_batched(xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Projects points using batched homography matrices.
    Args:
        xy: Points to be projected.
            Shape: (B, *, 2), where B is batch size
        homography: Batch of homography matrices.
            Shape: (B, 3, 3)
    Returns:
        Projected points.
        Shape: (B, *, 2)
    """

    # Reshape xy to (B, -1, 2) to flatten all middle dimensions.
    xy_reshaped = xy.reshape(xy.shape[0], -1, 2)

    # Homogeneous coordinates.
    ones_shape = xy_reshaped.shape[:-1] + (1,)
    ones = torch.ones(ones_shape, device=xy.device)
    xy_hom = torch.cat((xy_reshaped, ones), dim=-1)  # (B, -1, 3)

    # Project the points.
    xy_proj = torch.bmm(xy_hom, homography.transpose(1, 2))  # (B, -1, 3)

    # Euclidean coordinates
    xy_proj = xy_proj / xy_proj[..., 2:]
    xy_proj = xy_proj[..., :2]  # (B, -1, 2)

    # Restore original dimensions
    xy_proj = xy_proj.view(xy.shape)

    return xy_proj
