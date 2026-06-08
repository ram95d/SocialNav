"""Forecasting module. Provides an interface to use the trained model to
predict trajectories.

Provides the following functions:
- predict: Predicts the next pred_len values of observations using the model.
"""

import math
from typing import Literal, Optional

import torch
import numpy as np
import lightning as L
import scipy.signal

import rl_detect.model.datasets.traj_utils as traj_utils
import rl_detect.model.datasets.augmentation as augmentation
import rl_detect.model.model_utils as model_utils


# Observations per second used to train the model.
TRAIN_SCENE_FPS = 2.5       # 0.4 seconds


@torch.inference_mode()
def predict(model: L.LightningModule,
            scene_SO2: torch.Tensor,
            bev_frame_CHW: torch.Tensor, # TODO: potentially optional
            agent_ids: torch.Tensor,
            scene_fps: float,
            obs_len: int,
            pred_len: int,
            num_samples: int,
            noise_type: Optional[Literal['local', 'global']] = None,
            fixed_noise: Optional[bool] = None
            ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Predicts the next pred_len values of observations using the model.

    Moreover, returns the observations used to predict the next pred_len values,
    since they may be different from the input observations, due to the
    following reasons:
    - The model uses up to obs_len observations to perform the prediction.
    - A sampling operation is performed since the model needs the observations
        at a fixed interval.
    - The missing observations in the middle are interpolated.
    - The missing observations at the end are predicted using the model.

    All the observed trajectories with at least 2 observations (after sampling)
    are used in the prediction.

    Args:
        scene_SO2: Scene tensor.
            Shape: (scene_size, n_observations, 2).
        scene_fps: Scene fps.
            Must be a multiple of TRAIN_SCENE_FPS.
        obs_len: Number of steps to observe.
        pred_len: Number of steps to predict.
        num_samples: Number of samples to generate.

    Returns:
        Tuple of two tensors:
        - New scene tensor containing the next pred_len values.
            Shape: (scene_size, pred_len, 2).
        - A modified version of the input scene tensor, containing the
            observations used to predict the next pred_len values.
            Shape: (scene_size, seq_len, 2).
    """

    # Check if model supports sampling.
    sampling_info = None
    if num_samples > 1:
        sampling_info = model.sampling_info()
        if sampling_info is None:
            raise ValueError('Model does not support sampling')
        if noise_type is None:
            raise ValueError('noise_type must be specified')
        if fixed_noise is None:
            raise ValueError('fixed_noise must be specified')

    input_scene_size = scene_SO2.shape[0]

    # To float32.
    scene_SO2 = scene_SO2.float()

    # Compute scene sampling interval.
    scene_sampling_interval = scene_fps / TRAIN_SCENE_FPS
    # Ensure that the scene_fps is a multiple of TRAIN_SCENE_FPS.
    if not math.isclose(scene_sampling_interval,
                        round(scene_sampling_interval)):
        raise ValueError('scene_fps must be a multiple of TRAIN_SCENE_FPS')
    scene_sampling_interval = int(round(scene_sampling_interval))

    # Prepare the scene to be used by the model.
    new_scene_SO2 = _prepare_scene(scene_SO2,
                                   scene_sampling_interval,
                                   obs_len)

    # Compute the mask of the agents with enough observations.
    enough_obs_mask_S = traj_utils.check_enough_observations(new_scene_SO2)

    # Drop non visible pedestrians.
    # TODO: maybe move in prepare_scene
    visible_mask_S = ~torch.isnan(new_scene_SO2[:, -1, 0])

    # Combine the two masks.
    final_mask_S = enough_obs_mask_S & visible_mask_S

    # Ensure at least one agent has enough observations and is visible.
    if not torch.any(final_mask_S): return None

    # Filter out agents with not enough observations and non visible agents.
    new_scene_SO2 = new_scene_SO2[final_mask_S]
    new_scene_SO2 = new_scene_SO2.to(model.device)
    new_agent_ids = agent_ids[final_mask_S]

    # Smooth the scene.
    new_scene_SO2 = _smooth(new_scene_SO2)

    # Center the scene (so that data distribution matches training data).
    new_scene_centered_1SO2, barycenter_12 = \
        augmentation.center_scenes(new_scene_SO2[None], -1)
    new_scene_centered_SO2 = new_scene_centered_1SO2[0]

    ##############################################################

    # Create a (fake for now) segmentation mask from the BEV frame.
    bev_frame_CHW = bev_frame_CHW.to(new_scene_SO2.device)
    map_mask_1HW = (bev_frame_CHW.float() / 255.0).mean(dim=0, keepdim=True) #> 0.5
    map_mask_1HW = (map_mask_1HW * 255).to(torch.uint8)

    # `extract_patches` expects meters, and `new_scene_centered_SO2` is in meters.
    # `scene_transform_matrix` is the inverse of the centering transformation.
    scene_transform_matrix = torch.eye(3, device=new_scene_SO2.device)
    scene_transform_matrix[:2, 2] = barycenter_12[0]  # Inverse of centering translation

    # TODO: use the actual homography from meters to pixels, and rescale map_mask.
    # `homography_meters2mask` should scale from meters to pixels (1m = 100px).
    homography_meters2mask = torch.tensor([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]], device=new_scene_SO2.device)

    ##############################################################

    # Generate noise.
    if num_samples > 1:
        assert sampling_info is not None
        assert noise_type is not None
        assert fixed_noise is not None
        noise = _gen_noise(num_samples=num_samples,
                           noise_dim=sampling_info.noise_dim,
                           noise_distrib=sampling_info.noise_distrib,
                           noise_type=noise_type,
                           agent_ids=new_agent_ids,
                           fixed_noise=fixed_noise,
                           device=model.device)
    else:
        noise = None

    # Predict
    model.eval()
    # Return the predictions.
    # TODO: support map-based models
    scene_size = new_scene_centered_SO2.shape[0]
    # Repeat the transformation matrix for each agent in the scene
    scene_transform_matrix_B33 = scene_transform_matrix.unsqueeze(0).repeat(scene_size, 1, 1)
    homography_2mask_B33 = homography_meters2mask.unsqueeze(0).repeat(scene_size, 1, 1)

    preds = model.forward(new_scene_centered_SO2,
                          traj_gt_BP2=None,
                          scene_idx_B=torch.zeros(scene_size, dtype=torch.int),
                          map_mask_B1HW=map_mask_1HW.unsqueeze(0).repeat(scene_size, 1, 1, 1),
                          scene_transform_matrix_B33=scene_transform_matrix_B33,
                          homography_2mask_B33=homography_2mask_B33,
                          num_samples=num_samples,
                          noise_type=noise_type,
                          noise=noise)

    preds_SKP2 = preds['traj_pred_hat_BKP2']
    preds_KSP2 = preds_SKP2.permute(1, 0, 2, 3)

    # Undo the centering transformation.
    preds_KSP2 = augmentation.undo_center_scenes_batch(preds_KSP2, barycenter_12)

    # Output tensors, so that the output has the same scene_size as the input.
    output_preds_KSP2 = \
        torch.full((num_samples, input_scene_size, pred_len, 2), np.nan)
    output_obs_SO2 = \
        torch.full((input_scene_size, new_scene_SO2.shape[1], 2), np.nan)

    output_preds_KSP2[:, final_mask_S] = preds_KSP2.cpu()
    output_obs_SO2[final_mask_S] = new_scene_SO2.cpu()

    return output_preds_KSP2, output_obs_SO2


def _prepare_scene(scene: torch.Tensor,
                   scene_sampling_interval: int,
                   obs_len: int) -> torch.Tensor:
    """Prepares the scene to be used by the model.

    Performs the following steps:
    - Fills the missing observations by interpolating the existing ones.
    - Samples the observations at the given interval.

    Args:
        scene: Scene tensor.
            Shape: (scene_size, n_observations, 2).
        scene_sampling_interval: Scene sampling interval.
        obs_len: Number of steps to observe at the given interval.

    Returns:
        New scene tensor containing the observations sampled at the given
        interval and with missing observations in the middle interpolated.
        Missing observations at the beginning and at the end are left as nan,
        to make all the sequences have the same length as needed by a Tensor.
    """

    # For sure, the model doesn't need more than obs_len observations.
    max_obs = int((obs_len - 1) * scene_sampling_interval + 1)
    # Add a bit of margin to allow interpolating missing values
    # at the start, in case an older observation is known.
    max_obs += 10

    # Clip the sequence length to the maximum needed or available.
    # For sure, the model doesn't need more than obs_len observations.
    # And also the model doesn't need more than the available observations.
    max_obs = min(max_obs, scene.shape[1])

    # Start from empty tensor.
    # Shape: (scene_size, max_obs, 2)
    base = torch.full((scene.shape[0], max_obs, 2), np.nan)

    # Fill the tensor with the available observations,
    # starting from the end, since we know that the last
    # observations are the most recent. The first observations
    # are the oldest, and the missing ones are filled with nan as padding.
    base[:, -max_obs:] = scene[:, -max_obs:]

    # Interpolate the missing values in the middle.
    # Shape: (scene_size, max_obs, 2)
    traj_utils.fill_missing(base)

    # Size after sampling: 1 <= size_after_sampling <= obs_len.
    temp = int((scene.shape[1] - 1) // scene_sampling_interval + 1)
    size_after_sampling = min(temp, obs_len)

    # Sample the observations at the given interval.
    # Shape: (scene_size, size_after_sampling, 2)
    first_obs_idx = -(scene_sampling_interval * (size_after_sampling - 1) + 1)
    new_scene = base[:, first_obs_idx::scene_sampling_interval]

    return new_scene


def _smooth(scene: torch.Tensor) -> torch.Tensor:
    """Smooths the scene using Savitzky-Golay filter.

    Args:
        scene: Scene tensor.
            Shape: (scene_size, seq_len, 2).

    Returns:
        New scene tensor containing the smoothed values.
        Shape: (scene_size, seq_len, 2).
    """

    # Smooth the scene.
    window = 5

    # Convert to numpy.
    new_scene = scene.cpu().detach().numpy()

    # Mask of nan values.
    # Shape: (scene_size, seq_len)
    nan_mask = np.isnan(new_scene).any(axis=-1)

    # Iterate over the agents.
    for i in range(scene.shape[0]):
        # Mask of non nan values for the agent.
        # Shape: (seq_len,)
        mask = ~nan_mask[i]
        # Track of the agent.
        # Shape: (seq_len, 2)
        track = new_scene[i, mask]

        # Skip agents with not enough observations.
        if track.shape[0] <= window:
            continue

        # Smooth the track.
        new_scene[i, mask] = \
            scipy.signal.savgol_filter(track, window, 2, axis=0)

    # Convert back to Tensor.
    new_scene = torch.from_numpy(new_scene).float().to(scene)

    return new_scene


def _gen_noise(num_samples: int,
               noise_dim: int,
               noise_distrib: Literal['gaussian', 'uniform'],
               noise_type: Literal['local', 'global'],
               agent_ids: torch.Tensor,
               fixed_noise: bool,
               device) -> torch.Tensor:
    """Generates noise vectors.

    Args:
        num_samples: Number of samples to generate.
        noise_dim: Dimension of the noise vectors.
        noise_distrib: Distribution of the noise vectors.
            Must be 'gaussian' or 'uniform'.
        agent_ids: List of agent ids.
        fixed_noise: Whether to use fixed noise.

    Returns:
        Noise tensor.
            Shape: (num_samples, noise_dim).
    """

    K, S, L = num_samples, len(agent_ids), noise_dim

    noise_shape = (S, K, L) if noise_type == 'local' else (1, K, L)

    if not fixed_noise:
        return model_utils.gen_noise(noise_shape, noise_distrib, device=device)

    if noise_type == 'global':
        rng = torch.Generator(device=device)
        rng.manual_seed(0)

        return model_utils.gen_noise(noise_shape,
                                     noise_distrib,
                                     generator=rng,
                                     device=device)

    # Non-fixed local noise.
    noise_list = []
    for agent_id in agent_ids:
        agent_id = int(agent_id.item())
        # Use the agent id as seed.
        rng = torch.Generator(device=device)
        rng.manual_seed(agent_id)

        # Generate the noise for all the K samples of the agent.
        noise_KL = model_utils.gen_noise((K, L),
                                         noise_distrib,
                                         generator=rng,
                                         device=device)

        noise_list.append(noise_KL)

    return torch.stack(noise_list, dim=1)
