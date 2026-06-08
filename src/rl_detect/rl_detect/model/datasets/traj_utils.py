"""Utility functions for trajectory data processing."""

import torch


def check_enough_observations(scene: torch.Tensor,
                              min_obs: int = 2) -> torch.Tensor:
    """Checks if the sequences in the scene tensor have enough observations.

    At least min_obs observations per person are needed.

    Args:
        scene: The scene tensor (shape: (scene_size, seq_len, 2)).
        min_obs: Minimum number of observations needed. Default: 2.

    Returns:
        A boolean tensor of shape (scene_size,) where True means that the
        sequence has enough observations.
    """

    # True if point is not nan.
    # Shape: (scene_size, seq_len)
    not_nan_mask = (~torch.isnan(scene)).all(dim=-1)

    # Number of not nan points per sequence.
    # Shape: (scene_size,)
    obs_count = not_nan_mask.sum(dim=-1)

    # Return True if i-th sequence has at least min_obs observations.
    return obs_count >= min_obs


def check_visible(scene: torch.Tensor, last_obs_idx: int = -1) -> torch.Tensor:
    """Checks if the people in the scene tensor are visible.

    A person is visible if its last observation is not nan.
    The last observation is the one at index last_obs_idx.

    Args:
        scene: The scene tensor (shape: (scene_size, seq_len, 2))).
        last_obs_idx: The index of the observation to check. Default: -1
            (last observation, assuming only the observed part of the
            trajectory is provided).
            Useful when the scene tensor contains the full trajectory.

    Returns:
        A boolean tensor of shape (scene_size,) where True means that the
        person is visible.
    """

    return (~torch.isnan(scene[:, last_obs_idx, :])).all(dim=-1)


def keep_only_visible(scene: torch.Tensor, obs_len: int):
    # Check if all the pedestrians are visible in the scene.
    # A pedestrian is visible if the last observation
    # is not nan.
    last_obs_idx = obs_len - 1
    visible_mask = check_visible(scene, last_obs_idx)

    # Keep only the visible pedestrians.
    scene = scene[visible_mask]

    return scene


def drop_if_not_enough_observations(scene: torch.Tensor,
                                    obs_len: int,
                                    min_obs: int):
    # Check if all the pedestrians have at least min_obs observations.

    # True if point is not nan.
    # Shape: (batch_size, seq_len)
    not_nan_mask = (~torch.isnan(scene[:, :obs_len])).all(dim=-1)

    # Number of not nan points per sequence.
    # Shape: (batch_size,)
    obs_count = not_nan_mask.sum(dim=-1)

    # Return True if i-th sequence has at least min_obs observations.
    enough_obs = obs_count >= min_obs

    # Keep only the pedestrians with at least min_obs observations.
    return scene[enough_obs]


def drop_if_not_enough_predictions(scene: torch.Tensor,
                                   pred_len: int,
                                   min_pred: int):
    # Check if all the pedestrians have at least min_pred predictions.

    # True if point is not nan.
    # Shape: (batch_size, seq_len)
    not_nan_mask = (~torch.isnan(scene[:, -pred_len:])).all(dim=-1)

    # Number of not nan points per sequence.
    # Shape: (batch_size,)
    pred_count = not_nan_mask.sum(dim=-1)

    # Return True if i-th sequence has at least min_pred predictions.
    enough_pred = pred_count >= min_pred

    # Keep only the pedestrians with at least min_pred predictions.
    return scene[enough_pred]


def drop_if_unrealistic_step_size(scene_SA2: torch.Tensor,
                                  max_step_size: float):
    # Check if the step size between consecutive observations is realistic.

    # Compute the step size between consecutive observations.
    # Shape: (batch_size, seq_len - 1, 2)
    step_size_SA = scene_SA2.diff(1, dim=1).norm(dim=-1)

    # Return True if all the step sizes of the i-th sequence are realistic.
    realistic_step_size_S = (step_size_SA <= max_step_size).all(dim=-1)

    # Keep only the pedestrians with realistic step sizes.
    return scene_SA2[realistic_step_size_S]


def fill_missing(scene: torch.Tensor):
    """Fills the missing observations by interpolating the existing ones.

    Does not fill the missing observations at the beginning and at the end
    of the sequence, only the ones in the middle.

    For better results, provide the scene at high resolution, that is,
    before sampling the observations, so that the function can exploit
    the additional information.

    Modifies the input tensor in place.

    Args:
        scene: Scene tensor, containing nan for missing observations.
            Shape: (scene_size, seq_len, 2).
    """

    # For each visible pedestrian, interpolate the missing observations.
    for track in scene:
        # track: (seq_len, 2)

        # Find the missing observations.
        # Shape: (seq_len,)
        missing = torch.any(torch.isnan(track), dim=-1)

        # Find the index of all the non missing observations.
        # Shape: (num_non_missing,)
        non_missing = torch.where(~missing)[0]

        # List of pair of indeces of non-consecutive non missing observations.
        # Each pair represents an interval with at least one missing
        # observation, and the two indeces point to the first and last
        # non missing observations of the interval.
        intervals = []
        for i in range(1, len(non_missing)):
            # If there is a missing observation between the two non missing
            # observations, add the interval to the list.
            if non_missing[i] - non_missing[i-1] > 1:
                intervals.append((non_missing[i-1], non_missing[i]))

        # Interpolate the missing observations.
        for start, end in intervals:
            # start: index of the start of the missing interval.
            # end: index of the end of the missing interval.

            # Interpolate missing observations for each dimension separately.
            # Shape: (end - start - 1, 2)
            interpolated_values_x = \
                torch.linspace(track[start][0], track[end][0], end - start - 1)
            interpolated_values_y = \
                torch.linspace(track[start][1], track[end][1], end - start - 1)

            # Combine the interpolated values for each dimension.
            interpolated_values = torch.stack((interpolated_values_x,
                                               interpolated_values_y),
                                              dim=-1)

            # Fill the missing observations with the interpolated values.
            track[start+1:end] = interpolated_values
