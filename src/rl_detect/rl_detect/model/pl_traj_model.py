"""PyTorch Lightning models for trajectory prediction."""

from typing import Optional, Literal
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import EarlyStopping
import cv2 as cv

from rl_detect.model.pl_base import ConfigurableLitModule
import rl_detect.model.metrics as metrics
from .metrics import ADE, FDE, Collisions, EnvironmentCollisions
from .social_nce import (
    SocialNceLoss, ISocialNceCompatible, SocialQueryEmbedder, SocialKeyEmbedder
)
from .map_nce import (
    MapNceLoss, IMapNceCompatible, MapQueryEmbedder, MapKeyEmbedder
)
import rl_detect.utils as utils
from . import model_utils
from .sampling_info import SamplingInfo


class BaseTrajectoryLitModule(ConfigurableLitModule, ABC):
    """PyTorch Lightning model for trajectory prediction."""

    def __init__(self,
                 model: nn.Module,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 social_nce_loss_weight: float,
                 social_nce_temperature: Optional[float],
                 social_nce_proj_size: Optional[int],
                 map_nce_loss_weight: float,
                 map_nce_num_contour_points: Optional[int],
                 map_nce_temperature: Optional[float],
                 map_nce_proj_size: Optional[int],
                 env_collision_loss_weight: float,
                 goal_net_pretrain_epochs: int,
                 goal_loss_weight: float,
                 goal_matching_loss_weight: float,
                 goal_matching_loss_mode: Literal['best', 'all'],
                 optimizer: Optional[dict] = None,
                 lr_scheduler: Optional[dict] = None,
                 early_stopping: Optional[dict] = None,
                 gradient_clipping: Optional[dict] = None):
        """Builds the model.

        Args:
            model: The model to use for the prediction.
            obs_len: The length of the observed trajectory.
            pred_len: The length of the predicted trajectory.
            num_samples: The number of samples to generate for each scene,
                in the training phase (and also to compute validation and test
                metrics). At inference time, the model will generate as many
                samples as specified in the num_samples argument of the forward
                method.
            social_nce_loss_weight: The weight of the social NCE loss.
            social_nce_proj_size: The size of the projection for the social NCE
                loss.
            TODO
        """

        super().__init__(optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         early_stopping=early_stopping,
                         gradient_clipping=gradient_clipping)

        # Build the model with the given parameters.
        self.model = model

        # Number of observed and predicted points.
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_samples = num_samples

        # Goal losses.
        self.goal_net_pretrain_epochs = goal_net_pretrain_epochs
        self.goal_loss_weight = goal_loss_weight
        self.goal_matching_loss_weight = goal_matching_loss_weight
        self.goal_matching_loss_mode = goal_matching_loss_mode

        # Social NCE loss.
        self.social_nce_loss_weight = social_nce_loss_weight
        if self.social_nce_loss_weight > 0:
            if not isinstance(self.model, ISocialNceCompatible):
                raise ValueError('social_nce_loss_weight > 0 but the model is '
                                 'not compatible with the SocialNCE loss.')

            # Query and key projection heads.
            query_proj = SocialQueryEmbedder(model.social_encoding_size(),
                                            social_nce_proj_size)
            event_encoder = SocialKeyEmbedder(2, social_nce_proj_size)

            self.social_nce = SocialNceLoss(obs_len,
                                            pred_len,
                                            query_proj,
                                            event_encoder,
                                            social_nce_temperature)

        # Map NCE loss.
        self.map_nce_loss_weight = map_nce_loss_weight
        if self.map_nce_loss_weight > 0:
            if not isinstance(self.model, IMapNceCompatible):
                raise ValueError('map_nce_loss_weight > 0 but the model is '
                                 'not compatible with the MapNCE loss.')

            # Query and key projection heads.
            query_proj = MapQueryEmbedder(model.map_encoding_size(),
                                          map_nce_proj_size)
            event_encoder = MapKeyEmbedder(2, map_nce_proj_size)

            self.map_nce = MapNceLoss(obs_len,
                                      pred_len,
                                      map_nce_num_contour_points,
                                      query_proj,
                                      event_encoder,
                                      map_nce_temperature)

        # Environment collision loss.
        self.env_collision_loss_weight = env_collision_loss_weight

        # Metrics.
        self.train_ade = ADE(pred_len)
        self.train_fde = FDE()
        self.train_env_collisions = EnvironmentCollisions()
        self.train_col_pred = Collisions()
        self.train_col_gt = Collisions()
        self.val_ade = ADE(pred_len)
        self.val_fde = FDE()
        self.val_env_collisions = EnvironmentCollisions()
        self.val_col_pred = Collisions()
        self.val_col_gt = Collisions()
        self.test_ade = ADE(pred_len)
        self.test_fde = FDE()
        self.test_env_collisions = EnvironmentCollisions()
        self.test_col_pred = Collisions()
        self.test_col_gt = Collisions()

    @abstractmethod
    def loss(self,
             output: dict,
             traj_obs_BO2: Tensor,
             traj_pred_BP2: Tensor) -> Tensor:
        pass

    @abstractmethod
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
        """Forward pass through the model.

        This method should be implemented by the child class.
        So that each model can choose the arguments it needs.

        Args:
            scene_SO2: The scene tensor. Need at least 2 observed points.
                Shape: (scene_size, n_obs, 2).
            scene_transform_matrix: The transformation matrix for the scene.
                Useful for undoing data augmentation, so that the it is easy
                to align the trajectories with the map. Default: None.
                Shape: (3, 3).
            homography_2mask: Homography matrix from the scene to the mask.
                Useful for aligning the trajectories with the
                map. Default: None.
            num_samples: Number of samples to generate. Default: 1.
            noise_type: The type of noise to use.
                - 'local': Different noise for each pedestrian in a scene.
                - 'global': Same noise for all pedestrians in a scene.
                Default: 'local'.
            noise: The noise tensor. Used only if num_samples > 1.
                Valid shapes:
                    - (num_samples, scene_size, noise_dim)
                    - (num_samples, noise_dim).
                Default: None.

        Returns:
            TODO
            The predicted trajectory.
            Shape: (num_samples, scene_size, pred_len, 2).
        """

        pass

    def training_step(self, batch, batch_idx):
        """Training step."""

        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        metrics = self._shared_step(batch, batch_idx, 'val')

        # Log hyperparameter metrics for hyperparameters view in Tensorboard.
        hp_metrics = {
            "hp/ade": metrics['ade'],
            "hp/fde": metrics['fde'],
            # "hp/col_pred": metrics['col_pred'],
            # "hp/col_gt": metrics['col_gt'],
        }
        self.log_dict(hp_metrics,
                      prog_bar=False,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch[0].size(0))

        return metrics

    def test_step(self, batch, batch_idx):
        """Test step."""

        return self._shared_step(batch, batch_idx, 'test')

    def _shared_step(self, batch, batch_idx, split):
        """Shared step for training, validation and test."""

        # batch: bSA2 where S dimension might contain padding,
        # since the number of pedestrians in each scene is different.

        num_scenes = len(batch)

        # Extract data from batch.
        (traj_orig_bSA2,
         map_mask_b1HW,
         transform_matrix_b33,
         dataset_name_list,
         homographies_list,
         coord_system_list) = batch

        # Remove batch padding, and flatten the scenes in the batch,
        # so that all trajectories are in the B dimension.
        # (b, S, A, 2) -> (B=b*S, A, 2)
        pad_mask_bS = ~(torch.isnan(traj_orig_bSA2)).all(dim=(2, 3))
        traj_orig_BA2 = traj_orig_bSA2[pad_mask_bS]
        num_traj = traj_orig_BA2.size(0)

        # Keep track of the scene each trajectory belongs to.
        scene_sizes_b = pad_mask_bS.sum(dim=1)
        scene_idx_B = torch.repeat_interleave(scene_sizes_b)

        # Fuse the homographies of the scenes into a single tensor.
        # homographies_list contains a list of dictionaries, where the i-th
        # dictionary contains the homographies for the i-th scene.
        hom_orig2meters_B33 = torch.stack([ homographies_list[i]["orig2meters"]
                                            for i in scene_idx_B ])
        hom_meters2orig_B33 = torch.stack([ homographies_list[i]["meters2orig"]
                                            for i in scene_idx_B ])
        hom_meters2mask_B33 = torch.stack([ homographies_list[i]["meters2mask"]
                                            for i in scene_idx_B ])

        # Project the scene to meters.
        traj_BA2 = utils.project_batched(traj_orig_BA2, hom_orig2meters_B33)

        # Split the scene into observed and predicted trajectories.
        # traj_BA2 is always of length obs_len + pred_len.
        traj_obs_BO2 = traj_BA2[:, :self.obs_len]
        traj_pred_BP2 = traj_BA2[:, -self.pred_len:]
        # Keep a copy of the ground truth in the original coordinates.
        traj_pred_orig_BP2 = traj_orig_BA2[:, -self.pred_len:]

        # Prepare the map mask for the scenes in the batch.
        map_mask_B1HW = map_mask_b1HW[scene_idx_B]
        # Prepare the transformation matrix for the scenes in the batch.
        transform_matrix_B33 = transform_matrix_b33[scene_idx_B]

        # Predict the trajectories.
        output = self.forward(traj_obs_BO2,
                              traj_gt_BP2=traj_pred_BP2 if split == 'train' else None,
                              scene_idx_B=scene_idx_B,
                              map_mask_B1HW=map_mask_B1HW,
                              scene_transform_matrix_B33=transform_matrix_B33,
                              homography_2mask_B33=hom_meters2mask_B33,
                              num_samples=self.num_samples)

        if 'logits_BTA' in output:
            loss = self.loss(output, traj_obs_BO2, traj_pred_BP2)

            # TODO: think better
            # TODO: think better
            # TODO: think better
            # TODO: think better

            # Compute metrics.
            ade, fde, col_pred, col_gt, env_collisions = \
                (0, 0, 0, 0, 0)

            # ade = metrics.compute_ade(output['full_traj_BT2'][:, -self.pred_len:], traj_pred_BP2).mean()

            best_loss = 0
            social_nce_loss = 0
            map_nce_loss = 0
            env_collision_loss = 0
            goal_loss = 0
            goal_matching_loss = 0

            # Log metrics and losses.
            self._log(split,
                      ade=ade,
                      fde=fde,
                      col_pred=col_pred,
                      col_gt=col_gt,
                      env_collisions=env_collisions,
                      loss=loss,
                      best_loss=best_loss,
                      social_nce_loss=social_nce_loss,
                      map_nce_loss=map_nce_loss,
                      env_collision_loss=env_collision_loss,
                      goal_loss=goal_loss,
                      goal_matching_loss=goal_matching_loss,
                      batch_size=num_scenes)

        else:
            # Unpack the output dict.
            traj_pred_hat_BKP2 = output['traj_pred_hat_BKP2']
            goal_BK2 = output.get('goal_BK2')
            history_embedding_BH = output.get('history_embedding_BH')
            social_embedding_BH = output.get('social_embedding_BH')
            map_embedding_BH = output.get('map_embedding_BH')
            map_patch_B1HW = output.get('map_patch_B1HW')

            # Compute the loss.
            (best_loss,
             env_collision_loss,
             social_nce_loss,
             map_nce_loss,
             goal_loss,
             goal_matching_loss,
             best_ade_pred_hat_BP2,
             best_fde_pred_hat_BP2,
             env_collisions_Z) = self._compute_loss(
                traj_pred_hat_BKP2=traj_pred_hat_BKP2,
                traj_pred_BP2=traj_pred_BP2,
                traj_BA2=traj_BA2,
                goal_BK2=goal_BK2,
                scene_idx_B=scene_idx_B,
                social_embedding_BH=social_embedding_BH,
                map_embedding_BH=map_embedding_BH,
                map_mask_B1HW=map_mask_B1HW,
                map_patch_B1HW=map_patch_B1HW,
                transform_matrix_B33=transform_matrix_B33,
                homography_meters2mask_B33=hom_meters2mask_B33,
            )

            if self.current_epoch < self.goal_net_pretrain_epochs:
                loss = self.goal_loss_weight * goal_loss \
                       + self.goal_matching_loss_weight * goal_matching_loss
            else:
                loss = best_loss \
                       + self.env_collision_loss_weight * env_collision_loss \
                       + self.social_nce_loss_weight * social_nce_loss \
                       + self.map_nce_loss_weight * map_nce_loss \
                       + self.goal_loss_weight * goal_loss \
                       + self.goal_matching_loss_weight * goal_matching_loss

            # Update metrics.
            self._update_metrics(split=split,
                                 traj_pred_hat_BKP2=traj_pred_hat_BKP2,
                                 best_ade_pred_hat_BP2=best_ade_pred_hat_BP2,
                                 best_fde_pred_hat_BP2=best_fde_pred_hat_BP2,
                                 scene_pred_BP2=traj_pred_BP2,
                                 scene_pred_orig_BP2=traj_pred_orig_BP2,
                                 scene_idx_B=scene_idx_B,
                                 map_mask_B1HW=map_mask_B1HW,
                                 transform_matrix_B33=transform_matrix_B33,
                                 homography_meters2orig_B33=hom_meters2orig_B33,
                                 homography_meters2mask_B33=hom_meters2mask_B33,
                                 env_collisions_Z=env_collisions_Z)

            # Compute metrics.
            compute_env_col = env_collisions_Z is not None or split == 'test'
            ade, fde, col_pred, col_gt, env_collisions = \
                self._compute_metrics(split, env_col=compute_env_col)

            # Log metrics and losses.
            self._log(split,
                      ade=ade,
                      fde=fde,
                      col_pred=col_pred,
                      col_gt=col_gt,
                      env_collisions=env_collisions,
                      loss=loss,
                      best_loss=best_loss,
                      social_nce_loss=social_nce_loss,
                      map_nce_loss=map_nce_loss,
                      env_collision_loss=env_collision_loss,
                      goal_loss=goal_loss,
                      goal_matching_loss=goal_matching_loss,
                      batch_size=num_scenes)

        # Return the metrics.
        return {
            'loss': loss,
            'ade': ade,
            'fde': fde,
            'col_pred': col_pred,
            'col_gt': col_gt,
            'env_col': env_collisions,
            'goal_loss': goal_loss,
            'goal_matching_loss': goal_matching_loss,
        }

    def predict_step(self, batch, batch_idx):
        """Predict step."""

        # Extract data from batch.
        (traj_orig_bSA2,
         map_mask_b1HW,
         # transform_matrix_b33,
         dataset_name_list,
         homographies_list,
         coord_system_list) = batch

        # Remove batch padding, and flatten the scenes in the batch.
        # (b, S, A, 2) -> (B=b*S, A, 2)
        pad_mask_bS = ~(torch.isnan(traj_orig_bSA2)).all(dim=(2, 3))
        traj_orig_BA2 = traj_orig_bSA2[pad_mask_bS]

        # Keep track of the scene each trajectory belongs to.
        scene_sizes_b = pad_mask_bS.sum(dim=1)
        scene_idx_B = torch.repeat_interleave(scene_sizes_b)

        # Fuse the homographies of the scenes into a single tensor.
        hom_orig2meters_B33 = torch.stack([homographies_list[i]["orig2meters"]
                                          for i in scene_idx_B])
        hom_meters2orig_B33 = torch.stack([homographies_list[i]["meters2orig"]
                                          for i in scene_idx_B])
        hom_meters2mask_B33 = torch.stack([homographies_list[i]["meters2mask"]
                                          for i in scene_idx_B])

        # Project the scene to meters.
        traj_BA2 = utils.project_batched(traj_orig_BA2, hom_orig2meters_B33)

        # Split the scene into observed and predicted trajectories.
        traj_obs_BO2 = traj_BA2[:, :self.obs_len]
        traj_pred_BP2 = traj_BA2[:, -self.pred_len:]
        # Keep a copy of the ground truth in the original coordinates.
        traj_obs_orig_BO2 = traj_orig_BA2[:, :self.obs_len]
        traj_pred_orig_BP2 = traj_orig_BA2[:, -self.pred_len:]

        # Prepare the map mask and transform matrix for the scenes.
        transform_matrix_b33 = torch.eye(3, device=traj_orig_bSA2.device).\
                                     unsqueeze(0).\
                                     repeat(traj_orig_bSA2.size(0), 1, 1)
        map_mask_B1HW = map_mask_b1HW[scene_idx_B]
        transform_matrix_B33 = transform_matrix_b33[scene_idx_B]

        # Predict the trajectories.
        output = self.forward(
            traj_obs_BO2,
            traj_gt_BP2=None,
            scene_idx_B=scene_idx_B,
            map_mask_B1HW=map_mask_B1HW,
            scene_transform_matrix_B33=transform_matrix_B33,
            homography_2mask_B33=hom_meters2mask_B33,
            num_samples=self.num_samples
        )

        # Unpack the output dict.
        traj_pred_hat_BKP2 = output['traj_pred_hat_BKP2']

        # Transform the predicted trajectories to the original coordinates.
        traj_pred_hat_orig_BKP2 = utils.project_batched(
            traj_pred_hat_BKP2,
            hom_meters2orig_B33
        )

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: debug remove
        # # m = torch.flipud(map_mask_B1HW[0, 0])
        # m = map_mask_B1HW[0, 0]
        # map_rgb = cv.cvtColor(m.cpu().numpy(), cv.COLOR_GRAY2BGR)
        # # project the observed points to the mask
        # traj_obs_mask_BO2 = utils.project_batched(traj_obs_BO2, hom_meters2mask_B33)
        # # project the gt points to the mask
        # traj_pred_mask_BP2 = utils.project_batched(traj_pred_BP2, hom_meters2mask_B33)
        # # project the predicted points to the mask
        # traj_pred_hat_mask_BKP2 = utils.project_batched(traj_pred_hat_BKP2, hom_meters2mask_B33)
        # traj_obs_mask_BO2 = traj_obs_mask_BO2[:1]
        # traj_pred_mask_BP2 = traj_pred_mask_BP2[:1]
        # traj_pred_hat_mask_BKP2 = traj_pred_hat_mask_BKP2[:1]
        # # draw observed points
        # for i in range(traj_obs_mask_BO2.size(0)):
        #     for j in range(traj_obs_mask_BO2.size(1)):
        #         cv.circle(map_rgb, tuple(traj_obs_mask_BO2[i, j].int().tolist()), 1, (0, 0, 255), -1)
        # # draw gt points
        # for i in range(traj_pred_mask_BP2.size(0)):
        #     for j in range(traj_pred_mask_BP2.size(1)):
        #         cv.circle(map_rgb, tuple(traj_pred_mask_BP2[i, j].int().tolist()), 1, (255, 0, 0), -1)
        # # draw predicted points
        # for i in range(traj_pred_hat_mask_BKP2.size(0)):
        #     for j in range(traj_pred_hat_mask_BKP2.size(1)):
        #         for k in range(traj_pred_hat_mask_BKP2.size(2)):
        #             cv.circle(map_rgb, tuple(traj_pred_hat_mask_BKP2[i, j, k].int().tolist()), 1, (0, 255, 0), -1)
        # cv.imshow("observed", map_rgb)
        # cv.waitKey(0)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Reconstruct the batched scene structure.
        obs_traj_list = []
        pred_traj_list = []
        pred_hat_traj_list = []
        scene_name_list = []

        # Split predictions back into original scenes.
        start_idx = 0
        for i, scene_size in enumerate(scene_sizes_b):
            if scene_size == 0:
                continue

            end_idx = start_idx + scene_size

            obs_traj_list.append(traj_obs_orig_BO2[start_idx:end_idx])
            pred_traj_list.append(traj_pred_orig_BP2[start_idx:end_idx])
            pred_hat_traj_list.append(traj_pred_hat_orig_BKP2[start_idx:end_idx])
            scene_name_list.append(dataset_name_list[i])

            start_idx = end_idx

        # Return the predictions.
        return {
            'obs_trajs': obs_traj_list,
            'pred_trajs': pred_traj_list,
            'pred_hat_trajs': pred_hat_traj_list,
            'scene_names': scene_name_list,
        }

    def _compute_loss(self,
                      traj_pred_hat_BKP2: Tensor,
                      traj_pred_BP2: Tensor,
                      traj_BA2: Tensor,
                      goal_BK2: Optional[Tensor],
                      scene_idx_B: Tensor,
                      social_embedding_BH: Tensor,
                      map_embedding_BH: Tensor,
                      map_mask_B1HW: Tensor,
                      map_patch_B1HW: Tensor,
                      transform_matrix_B33: Tensor,
                      homography_meters2mask_B33: Tensor,
                      ) -> tuple[Tensor, Tensor, Tensor, Tensor,
                                 Tensor, Tensor, Tensor, Tensor,
                                 Optional[Tensor]]:
        """Compute the loss for a batch of scenes."""

        batch_size = traj_BA2.size(0)

        # Best ADE sample index for computing the Best-of-20 ADE metric.
        # Tensor of shape (batch_size,) with the index of the closest (in terms
        # of ADE) sample to the ground truth for each pedestrian.
        best_ade_sample_index_B = \
            model_utils.closest_sample_index(traj_pred_hat_BKP2,
                                             traj_pred_BP2,
                                             metric='ade')

        # Best FDE sample index for computing the Best-of-20 FDE metric.
        best_fde_sample_index_B = \
            model_utils.closest_sample_index(traj_pred_hat_BKP2,
                                             traj_pred_BP2,
                                             metric='fde')

        best_ade_pred_hat_BP2 = traj_pred_hat_BKP2[torch.arange(batch_size),
                                                   best_ade_sample_index_B]
        best_fde_pred_hat_BP2 = traj_pred_hat_BKP2[torch.arange(batch_size),
                                                   best_fde_sample_index_B]

        # Goal related losses.
        goal_loss = torch.tensor(0.0, requires_grad=True, device=traj_pred_hat_BKP2.device)
        goal_matching_loss = torch.tensor(0.0, requires_grad=True, device=traj_pred_hat_BKP2.device)
        if goal_BK2 is not None:
            # Best goal.
            best_goal_sample_index_B = \
                model_utils.closest_sample_index(goal_BK2,
                                                 traj_pred_BP2[:, -1],
                                                 metric='goal')
            best_goal_hat_B2 = goal_BK2[torch.arange(batch_size),
                                        best_goal_sample_index_B]
            goal_gt_B2 = traj_pred_BP2[:, -1]
            goal_loss = metrics.compute_fde(best_goal_hat_B2[:, None, :],
                                            goal_gt_B2[:, None, :]).mean()

            # Goal matching loss.
            if self.goal_matching_loss_mode == 'best':
                # Make the best prediction be close to its associated goal.
                best_ade_pred_hat_goal_B2 = goal_BK2[torch.arange(batch_size),
                                                     best_ade_sample_index_B]
                best_ade_pred_hat_goal_B2 = best_ade_pred_hat_goal_B2.detach()
                goal_matching_loss = (best_ade_pred_hat_goal_B2 - best_ade_pred_hat_BP2[:, -1]).norm(dim=-1).mean()
            elif self.goal_matching_loss_mode == 'all':
                # Make all predicted samples be close to their associated goals.
                goal_detach_BK2 = goal_BK2.detach()
                goal_matching_loss = (goal_detach_BK2 - traj_pred_hat_BKP2[:, :, -1]).norm(dim=-1).mean()
            else:
                raise ValueError(f'Unknown goal_matching_loss_mode: {self.goal_matching_loss_mode}')

        # Z = B * K
        env_collisions_Z = None
        env_collision_loss = torch.tensor(0.0, requires_grad=True, device=traj_pred_hat_BKP2.device)
        if self.env_collision_loss_weight > 0:
            # Compute the environmental collisions on all the samples.
            traj_pred_hat_ZP2 = traj_pred_hat_BKP2.view(-1, self.pred_len, 2)

            env_collisions_Z = model_utils.check_env_collisions(
                traj_pred_hat_ZP2,
                map_mask_B1HW,
                transform_matrix_B33,
                homography_meters2mask_B33,
            )
            env_collisions_BK = env_collisions_Z.view(batch_size,
                                                      self.num_samples)

            # Make the colliding trajectories be part of the loss.
            # Get the index of the ground truth trajectory for each
            # trajectory that collides with the environment.
            # Shape: (num_collisions,) where num_collisions <= Z
            gt_index_Z, _ = torch.where(env_collisions_BK)
            # Get the ground truth trajectory for each colliding trajectory.
            # Shape: (num_collisions, pred_len, 2)
            gt_ZP2 = traj_pred_BP2[gt_index_Z]

            # Compute the loss for the colliding trajectories (if any).
            if traj_pred_hat_ZP2[env_collisions_Z].size(0) > 0:
                env_collision_loss = \
                    metrics.compute_ade(traj_pred_hat_ZP2[env_collisions_Z], gt_ZP2).mean()

                if torch.isnan(env_collision_loss):
                    env_collision_loss = torch.tensor(0.0, requires_grad=True, device=env_collision_loss.device)

        # Compute variety loss (ADE).
        best_loss = metrics.compute_ade(best_ade_pred_hat_BP2, traj_pred_BP2).mean()

        # NCE losses.
        social_nce_loss = torch.tensor(0.0, device=traj_pred_hat_BKP2.device)
        map_nce_loss = torch.tensor(0.0, device=traj_pred_hat_BKP2.device)
        if self.trainer.training:
            # Compute the social NCE loss.
            if self.social_nce_loss_weight > 0:
                social_nce_loss = self.social_nce(traj_BA2,
                                                  social_embedding_BH,
                                                  scene_idx_B)

            # Compute the map NCE loss.
            if self.map_nce_loss_weight > 0:
                map_nce_loss = self.map_nce(traj_BA2,
                                            map_embedding_BH,
                                            map_patch_B1HW)

        # Return the losses and the closest predicted trajectory.
        return (best_loss,
                env_collision_loss,
                social_nce_loss,
                map_nce_loss,
                goal_loss,
                goal_matching_loss,
                best_ade_pred_hat_BP2,
                best_fde_pred_hat_BP2,
                env_collisions_Z)


    def _compute_scene_loss(self,
                            scene_pred_hat_KSP2: torch.Tensor,
                            scene_pred_SP2: torch.Tensor,
                            scene_SA2: torch.Tensor,
                            social_embeddings_SH: torch.Tensor,
                            map_embeddings_SH: torch.Tensor,
                            map_mask_1HW: torch.Tensor,
                            map_patches_S1HW: torch.Tensor,
                            transform_matrix: torch.Tensor,
                            homography_meters2mask: torch.Tensor,
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor,
                                       torch.Tensor]:
        # TODO: deprecated

        """Compute the loss for a scene.

        Args:
            scene_pred_hat_KSP2: The predicted trajectories.
                Assumed to be in meters.
                Shape: (num_samples, scene_size, pred_len, 2).
            scene_pred_SP2: The ground truth trajectories.
                Assumed to be in meters.
                Shape: (scene_size, pred_len, 2).
            scene_SA2: The scene tensor.
                Assumed to be in meters.
                Shape: (scene_size, obs_len + pred_len, 2).
            social_embeddings_SH: The social embeddings.
                Shape: (scene_size, social_embedding_size).
            map_embeddings_SH: The map embeddings.
                Shape: (scene_size, map_embedding_size).
            map_mask_1HW: The map mask.
                Shape: (1, H, W).
            map_patches_S1HW: The map patches.
                Shape: (scene_size, 1, H, W).
            transform_matrix: The transformation matrix for the scene.
                Shape: (3, 3).
            dataset_name: The name of the dataset.
            homography_meters2mask: The homography matrix from meters to mask.
                Shape: (3, 3).
        """

        scene_size = scene_SA2.size(0)

        # Variety loss.
        # Tensor of shape (scene_size,) with the index of the closest (in terms
        # of ADE) sample to the ground truth for each pedestrian.
        best_ade_sample_index_S = \
            model_utils.closest_sample_index(scene_pred_hat_KSP2,
                                       scene_pred_SP2,
                                       metric='ade')
        # Best FDE sample index for computing the Best-of-20 FDE metric.
        best_fde_sample_index_S = \
            model_utils.closest_sample_index(scene_pred_hat_KSP2,
                                       scene_pred_SP2,
                                       metric='fde')

        best_ade_pred_hat_SP2 = scene_pred_hat_KSP2[best_ade_sample_index_S,
                                                    torch.arange(scene_size)]
        best_fde_pred_hat_SP2 = scene_pred_hat_KSP2[best_fde_sample_index_S,
                                                    torch.arange(scene_size)]

        env_collisions_B = None
        env_collision_loss = torch.tensor(0.0, device=scene_pred_hat_KSP2.device)
        if self.env_collision_loss_weight > 0:
            # Compute the environmental collisions on all the samples.
            scene_pred_hat_BP2 = scene_pred_hat_KSP2.view(-1, self.pred_len, 2)
            env_collisions_B = model_utils.check_env_collisions(
                scene_pred_hat_BP2,
                map_mask_1HW,
                transform_matrix,
                homography_meters2mask,
            )
            env_collisions_KS = env_collisions_B.view(self.num_samples,
                                                      scene_size)

            # Make the colliding trajectories be part of the loss.
            # Get the index of the ground truth trajectory for each
            # trajectory that collides with the environment.
            # Shape: (num_collisions,) where num_collisions <= B
            _, gt_index_B = torch.where(env_collisions_KS)
            # Get the ground truth trajectory for each colliding trajectory.
            # Shape: (num_collisions, pred_len, 2)
            gt_BP2 = scene_pred_SP2[gt_index_B]

            # Compute the loss for the colliding trajectories (if any).
            if scene_pred_hat_BP2[env_collisions_B].size(0) > 0:
                env_collision_loss = \
                    F.mse_loss(scene_pred_hat_BP2[env_collisions_B],
                            gt_BP2,
                            reduction='sum')

                if torch.isnan(env_collision_loss):
                    env_collision_loss = torch.tensor(0.0, device=env_collision_loss.device)

        # Compute MSE loss.
        # Reduction is 'sum' to give the same weight to all the trajectories.
        # In fact, if we use 'mean', the weight given to trajectories
        # of different scenes would be different, because the number of
        # trajectories in each scene is different.
        mse_loss = F.mse_loss(best_ade_pred_hat_SP2,
                              scene_pred_SP2,
                              reduction='sum')

        social_nce_loss = 0
        map_nce_loss = 0
        if self.trainer.training:
            # Compute the social NCE loss.
            if self.social_nce_loss_weight > 0:
                social_nce_loss = self.social_nce(scene_SA2,
                                                  social_embeddings_SH)

            # Compute the map NCE loss.
            if self.map_nce_loss_weight > 0:
                map_nce_loss = self.map_nce(scene_SA2,
                                            map_embeddings_SH,
                                            map_patches_S1HW)

        # Return the losses and the closest predicted trajectory.
        return (mse_loss,
                env_collision_loss,
                social_nce_loss,
                map_nce_loss,
                best_ade_pred_hat_SP2,
                best_fde_pred_hat_SP2,
                env_collisions_B)

    def _update_metrics(self,
                        split: Literal['train', 'val', 'test'],
                        traj_pred_hat_BKP2: torch.Tensor,
                        best_ade_pred_hat_BP2: torch.Tensor,
                        best_fde_pred_hat_BP2: torch.Tensor,
                        scene_pred_BP2: torch.Tensor,
                        scene_pred_orig_BP2: torch.Tensor,
                        scene_idx_B: torch.Tensor,
                        map_mask_B1HW: torch.Tensor,
                        transform_matrix_B33: torch.Tensor,
                        homography_meters2orig_B33: torch.Tensor,
                        homography_meters2mask_B33: torch.Tensor,
                        env_collisions_Z: torch.Tensor | None = None):

        # Compute the predicted trajectory in the original coordinates.
        best_ade_pred_hat_orig_BP2 = \
            utils.project_batched(best_ade_pred_hat_BP2, homography_meters2orig_B33)
        best_fde_pred_hat_orig_BP2 = \
            utils.project_batched(best_fde_pred_hat_BP2, homography_meters2orig_B33)

        # Merge the batch dimension with the samples dimension.
        scene_pred_hat_ZP2 = traj_pred_hat_BKP2.view(-1, self.pred_len, 2)

        # Get the methods to compute the metrics.
        ade_method = getattr(self, f'{split}_ade')
        fde_method = getattr(self, f'{split}_fde')
        col_pred_method = getattr(self, f'{split}_col_pred')
        col_gt_method = getattr(self, f'{split}_col_gt')
        env_collisions_method = getattr(self, f'{split}_env_collisions')

        # Compute ADE/FDE metrics in the original coordinates.
        ade_method(best_ade_pred_hat_orig_BP2, scene_pred_orig_BP2)
        fde_method(best_fde_pred_hat_orig_BP2, scene_pred_orig_BP2)

        # Compute other metrics in the meters coordinates.
        col_pred_method(best_ade_pred_hat_BP2, scene_idx_B=scene_idx_B)
        col_gt_method(best_ade_pred_hat_BP2, scene_pred_BP2, scene_idx_B=scene_idx_B)

        # Compute environmental collisions only if collisions already computed,
        # or if we are in the test phase.
        if env_collisions_Z is not None or split == 'test':
            env_collisions_method(scene_pred_hat_ZP2,
                                  map_mask_B1HW,
                                  transform_matrix_B33,
                                  homography_meters2mask_B33,
                                  env_collisions_Z)

    def _compute_metrics(self, split, env_col=False):
        ade = getattr(self, f'{split}_ade')
        fde = getattr(self, f'{split}_fde')
        col_pred = getattr(self, f'{split}_col_pred')
        col_gt = getattr(self, f'{split}_col_gt')

        if env_col:
            env_collisions = getattr(self, f'{split}_env_collisions')
        else:
            env_collisions = 100

        return ade, fde, col_pred, col_gt, env_collisions

    def _log(self,
             split,
             ade,
             fde,
             col_pred,
             col_gt,
             env_collisions,
             loss,
             best_loss,
             social_nce_loss,
             map_nce_loss,
             env_collision_loss,
             goal_loss,
             goal_matching_loss,
             batch_size):
        # Log metrics and losses.
        log_dict_show = {
            f'{split}_ade': ade,
            f'{split}_fde': fde,
        }
        self.log_dict(log_dict_show,
                      prog_bar=True,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_size)

        # TODO: log nce loss only during training

        log_dict_hide = {
            f'{split}_loss': loss,
            f'{split}_best_loss': best_loss,

            f'{split}_social_nce_loss': social_nce_loss,
            f'{split}_map_nce_loss': map_nce_loss,

            f'{split}_col_pred': col_pred,
            f'{split}_col_gt': col_gt,

            f'{split}_env_collisions': env_collisions,
            f'{split}_env_collision_loss': env_collision_loss,

            f'{split}_goal_loss': goal_loss,
            f'{split}_goal_matching_loss': goal_matching_loss,
        }
        self.log_dict(log_dict_hide,
                      prog_bar=False,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_size)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            "hp/ade": 5,
            "hp/fde": 5,
            # "hp/col_pred": 1,
            # "hp/col_gt": 1,
        })

    @abstractmethod
    def sampling_info(self) -> SamplingInfo | None:
        """Check if the model supports sampling, and return the sampling info.

        Returns:
            Information about the sampling capabilities of the model,
            or None if the model does not support sampling.
        """

        pass
