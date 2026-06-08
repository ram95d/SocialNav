import torch
import torchmetrics

import rl_detect.model.model_utils as model_utils


def compute_ade(preds_BP2, target_BP2):
    """Compute the average displacement error (ADE) for each trajectory.

    Args:
        preds_BP2: Predicted trajectory, shape (*, pred_len, 2).
        target_BP2: Target trajectory, shape (*, pred_len, 2).

    Returns:
        ADE for each trajectory in the batch, shape (*).
    """

    # Average displacement error (ADE) for each trajectory in the batch.
    return torch.norm(preds_BP2 - target_BP2, dim=-1).mean(dim=-1)


def compute_fde(preds_BP2, target_BP2):
    """Compute the final displacement error (FDE) for each trajectory.

    Args:
        preds_BP2: Predicted trajectory, shape (*, pred_len, 2).
        target_BP2: Target trajectory, shape (*, pred_len, 2).

    Returns:
        FDE for each trajectory in the batch, shape (*).
    """

    # Final displacement error (FDE) for each trajectory in the batch.
    return torch.norm(preds_BP2[..., -1, :] - target_BP2[..., -1, :], dim=-1)


# TODO: maybe merge into DisplacementError

class ADE(torchmetrics.Metric):
    def __init__(self, pred_len):
        super().__init__()
        self.add_state("ade_sum",
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.pred_len = pred_len

    def update(self, preds, target):
        # preds: (batch, seq_len, 2)
        # Displacement error for each point in the trajectory.
        ade_B = compute_ade(preds[:, -self.pred_len:],
                            target[:, -self.pred_len:])
        # Sum of all ADEs in the batch.
        ade_sum = ade_B.sum()

        # Update the ade_sum state.
        self.ade_sum += ade_sum
        # Number of trajectories in the batch.
        self.count += target.size(0)

    def compute(self):
        return self.ade_sum / self.count


class FDE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("fde_sum",
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds, target):
        # preds: (batch, seq_len, 2)
        # Displacement error for last point in the trajectory.
        fde_B = compute_fde(preds, target)
        # Sum of all FDEs in the batch.
        fde_sum = fde_B.sum()

        # Update the fde_sum state.
        self.fde_sum += fde_sum
        # Number of trajectories in the batch.
        self.count += target.size(0)

    def compute(self):
        return self.fde_sum / self.count


class Collisions(torchmetrics.Metric):
    def __init__(self, person_radius=0.1, parts=1):
        super().__init__()

        self.person_radius = person_radius
        self.parts = parts

        self.add_state("collision_avg_sum",
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, pred_BP2, target_BP2=None, scene_idx_B=None):
        if scene_idx_B is None:
            raise ValueError("scene_idx_B is required for batched scene processing")

        # Augment the trajectory resolution
        pred_BT2 = model_utils.augment_traj_resolution(pred_BP2, self.parts)
        if target_BP2 is None:
            target_BT2 = pred_BT2
        else:
            target_BT2 = model_utils.augment_traj_resolution(target_BP2, self.parts)

        # Compute pointwise distances between all pairs of trajectories
        diff_BBT2 = pred_BT2.unsqueeze(1) - target_BT2.unsqueeze(0)
        distances_BBT = torch.norm(diff_BBT2, dim=-1)

        # Create scene comparison mask (B x B)
        # True where trajectories are from the same scene
        scene_mask_BB = (scene_idx_B.unsqueeze(1) == scene_idx_B.unsqueeze(0))

        # Mask out self-collisions and trajectories from different scenes
        mask_BB = torch.eye(pred_BT2.size(0),
                          dtype=torch.bool,
                          device=pred_BP2.device)
        invalid_pairs_BB = ~scene_mask_BB | mask_BB
        distances_BBT[invalid_pairs_BB] = torch.inf

        # Check for collisions at each time step
        collision_BB = (distances_BBT <= 2 * self.person_radius).any(dim=-1)

        # Count collisions only within same scene
        collision_count_B = collision_BB.sum(dim=1)

        # Get number of trajectories per scene for normalization
        scene_sizes = torch.bincount(scene_idx_B)
        max_col_counts = scene_sizes[scene_idx_B] - 1  # subtract 1 for self
        max_col_counts = torch.clamp(max_col_counts, min=1)  # avoid division by zero

        # Compute collision average
        collision_avg_B = collision_count_B / max_col_counts

        # Update metrics
        self.collision_avg_sum += collision_avg_B.sum()
        self.count += pred_BP2.size(0)

    def compute(self):
        return self.collision_avg_sum / self.count * 100


class EnvironmentCollisions(torchmetrics.Metric):
    """Percentage of trajectories that collide with the environment."""

    def __init__(self):
        super().__init__()
        self.add_state("collision_sum",
                       default=torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self,
               pred_ZP2,
               map_mask_B1HW,
               scene_transform_matrix_B33,
               homography_meters2mask_B33,
               env_collisions_Z=None):
        """Update the metric with a batch of predicted trajectories.

        Args:
            pred_BP2: Predicted trajectories, shape (*, pred_len, 2).
            map_mask_1HW: The map mask, shape (1, H, W).
            scene_transform_matrix: The transformation matrix used
                in data augmentation, useful to undo the transformation.
            homography_meters2mask: The homography matrix from meters to the
                map mask.
            env_collisions_B: Optional tensor of shape (*), indicating
                whether each trajectory collides with the environment.
                Useful to avoid recomputing the collisions.
        """

        if env_collisions_Z is None:
            # Check for collisions.
            env_collisions_Z = model_utils.check_env_collisions(
                pred_ZP2,
                map_mask_B1HW,
                scene_transform_matrix_B33,
                homography_meters2mask_B33
            )
        self.collision_sum += env_collisions_Z.sum()

        # Number of trajectories in the batch.
        self.count += pred_ZP2.size(0)

    def compute(self):
        return self.collision_sum / self.count * 100
