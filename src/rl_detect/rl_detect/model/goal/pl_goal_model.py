"""Goal sampling network."""

from typing import Optional, Literal
from abc import ABC, abstractmethod
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt

import rl_detect.utils as utils
import model.model_utils as model_utils
from model.pl_base import ConfigurableLitModule
from model import metrics


logger = logging.getLogger(__name__)


class BaseGoalNetLitModule(ConfigurableLitModule, ABC):
    def __init__(self,
                 model: nn.Module,
                 obs_len: int,
                 pred_len: int,
                 num_samples: int,
                 optimizer: Optional[dict] = None,
                 lr_scheduler: Optional[dict] = None,
                 early_stopping: Optional[dict] = None,
                 gradient_clipping: Optional[dict] = None):

        super().__init__(optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         early_stopping=early_stopping,
                         gradient_clipping=gradient_clipping)

        # Model.
        self.model = model

        # Prediction parameters.
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_samples = num_samples

        # Losses.

        # Metrics.
        self.train_fde = metrics.FDE()
        self.train_env_collisions = metrics.EnvironmentCollisions()
        self.val_fde= metrics.FDE()
        self.val_env_collisions = metrics.EnvironmentCollisions()
        self.test_fde= metrics.FDE()
        self.test_env_collisions = metrics.EnvironmentCollisions()

    @abstractmethod
    def forward(self,
                traj_BO2: Tensor,
                map_mask_B1HW: Tensor,
                hom_meters2mask_B33: Tensor,
                hom_mask2meters_B33: Tensor
                ) -> dict:
        pass

    @abstractmethod
    def loss(self,
             output: dict,
             gt_goal_B2: Tensor,
             traj_obs_BO2: Tensor,
             hom_meters2mask_B33) -> Tensor:
        pass

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def _shared_step(self, batch, batch_idx, phase):
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
        hom_mask2meters_B33 = torch.stack([ homographies_list[i]["mask2meters"]
                                            for i in scene_idx_B ])

        # Project the scene to meters.
        traj_BA2 = utils.project_batched(traj_orig_BA2, hom_orig2meters_B33)

        # Split the scene into observed and predicted trajectories.
        # traj_BA2 is always of length obs_len + pred_len.
        traj_obs_BO2 = traj_BA2[:, :self.obs_len]
        traj_pred_BP2 = traj_BA2[:, -self.pred_len:]
        # Keep a copy of the ground truth in the original coordinates.
        traj_pred_orig_BP2 = traj_orig_BA2[:, -self.pred_len:]

        # Ground truth goals.
        gt_goal_B2 = traj_pred_BP2[:, -1]
        gt_goal_orig_B2 = traj_pred_orig_BP2[:, -1]

        # Prepare the map mask for the scenes in the batch.
        map_mask_B1HW = map_mask_b1HW[scene_idx_B]
        map_mask_B1HW = map_mask_B1HW / 255.0
        # Prepare the transformation matrix for the scenes in the batch.
        transform_matrix_B33 = transform_matrix_b33[scene_idx_B]

        # Compute goals.
        output = self.forward(traj_obs_BO2, map_mask_B1HW, hom_meters2mask_B33, hom_mask2meters_B33)

        # Extract the goals.
        goal_BK2 = output["goal_BK2"]
        goal_orig_BK2 = utils.project_batched(goal_BK2, hom_meters2orig_B33)

        best_goal_sample_index_B = \
            model_utils.closest_sample_index(goal_BK2,
                                             gt_goal_B2,
                                             metric='goal')
        best_goal_orig_B2 = goal_orig_BK2[torch.arange(num_traj), best_goal_sample_index_B]

        # Compute loss.
        loss = self.loss(output, gt_goal_B2, traj_obs_BO2, hom_meters2mask_B33)

        # TODO: maybe abstract `debug` method

        # Compute metrics.
        self._update_metrics(phase, best_goal_orig_B2, gt_goal_orig_B2)
        fde, env_col = self._compute_metrics(phase)

        # Log metrics.
        self._log(split=phase,
                  loss=loss,
                  fde=fde,
                  env_col=env_col,
                  batch_size=num_traj)

        return {
            'loss': loss,
            'fde': fde,
            'env_col': env_col,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step."""

        # Extract data from batch.
        (traj_orig_bSA2,
         map_mask_b1HW,
         # transform_matrix_b33,
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
        hom_mask2meters_B33 = torch.stack([ homographies_list[i]["mask2meters"]
                                            for i in scene_idx_B ])

        # Project the scene to meters.
        traj_BA2 = utils.project_batched(traj_orig_BA2, hom_orig2meters_B33)

        # Split the scene into observed and predicted trajectories.
        # traj_BA2 is always of length obs_len + pred_len.
        traj_obs_BO2 = traj_BA2[:, :self.obs_len]
        traj_pred_BP2 = traj_BA2[:, -self.pred_len:]
        # Keep a copy of the ground truth in the original coordinates.
        traj_obs_orig_BO2 = traj_orig_BA2[:, :self.obs_len]
        traj_pred_orig_BP2 = traj_orig_BA2[:, -self.pred_len:]

        # Ground truth goals.
        gt_goal_B2 = traj_pred_BP2[:, -1]
        gt_goal_orig_B2 = traj_pred_orig_BP2[:, -1]

        # Prepare the map mask for the scenes in the batch.
        map_mask_B1HW = map_mask_b1HW[scene_idx_B]
        map_mask_B1HW = map_mask_B1HW / 255.0

        # Compute goals.
        output = self.forward(traj_obs_BO2, map_mask_B1HW, hom_meters2mask_B33, hom_mask2meters_B33)

        # Extract the goals.
        goal_BK2 = output["goal_BK2"]
        goal_orig_BK2 = utils.project_batched(goal_BK2, hom_meters2orig_B33)

        best_goal_sample_index_B = \
            model_utils.closest_sample_index(goal_BK2,
                                             gt_goal_B2,
                                             metric='goal')
        best_goal_B2 = goal_BK2[torch.arange(num_traj), best_goal_sample_index_B]
        best_goal_orig_B2 = goal_orig_BK2[torch.arange(num_traj), best_goal_sample_index_B]

        # TODO: maybe abstract `debug` method

        #####################
        #####################

        # Display map, observed and predicted trajectories, and goals.
        # 5 random trajectories.
        import cv2 as cv
        n = 5
        idx = torch.randperm(num_traj)[:n]
        for i in idx:
            # First print fde (original coordinates).
            fde = metrics.compute_fde(best_goal_orig_B2[None, None, i], gt_goal_orig_B2[None, None, i])
            print(f"FDE: {fde.item()}")

            # Get the scene name.
            scene_name = dataset_name_list[scene_idx_B[i]]

            # Get the map.
            map_rgb = map_mask_B1HW[i,0].cpu().numpy()
            map_rgb = cv.cvtColor(map_rgb, cv.COLOR_GRAY2BGR)

            # Project the observed, future and goals to the mask.
            traj_obs_mask_O2 = utils.project(traj_obs_BO2[i], hom_meters2mask_B33[i])
            traj_pred_mask_P2 = utils.project(traj_pred_BP2[i], hom_meters2mask_B33[i])
            goals_mask_K2 = utils.project(goal_BK2[i], hom_meters2mask_B33[i])
            best_goal_mask_2 = utils.project(best_goal_B2[i], hom_meters2mask_B33[i])

            # Draw observed points.
            for j in range(traj_obs_mask_O2.size(0)):
                cv.circle(map_rgb,
                          tuple(traj_obs_mask_O2[j].int().tolist()),
                          1,
                          (0, 0, 0),
                          -1)

            # Draw future points.
            for j in range(traj_pred_mask_P2.size(0)):
                cv.circle(map_rgb,
                          tuple(traj_pred_mask_P2[j].int().tolist()),
                          1,
                          (0, 255, 0),
                          -1)

            # Draw goals.
            for j in range(goals_mask_K2.size(0)):
                cv.circle(map_rgb,
                          tuple(goals_mask_K2[j].int().tolist()),
                          1,
                          (0, 0, 255),
                          -1)

            # Draw best goal.
            cv.circle(map_rgb,
                      tuple(best_goal_mask_2.int().tolist()),
                      2,
                      (255, 0, 255),
                      -1)

            # Display.
            cv.imshow(scene_name, map_rgb)
            cv.waitKey(0)


        #####################
        #####################

        # Reconstruct the batched scene structure.
        obs_traj_list = []
        pred_traj_list = []
        goal_hat_list = []
        scene_name_list = []

        # Split predictions back into original scenes.
        start_idx = 0
        for i, scene_size in enumerate(scene_sizes_b):
            if scene_size == 0:
                continue

            end_idx = start_idx + scene_size

            obs_traj_list.append(traj_obs_orig_BO2[start_idx:end_idx])
            pred_traj_list.append(traj_pred_orig_BP2[start_idx:end_idx])
            goal_hat_list.append(goal_orig_BK2[start_idx:end_idx])
            scene_name_list.append(dataset_name_list[i])

            start_idx = end_idx

        # Return the predictions.
        return {
            'obs_trajs': obs_traj_list,
            'pred_trajs': pred_traj_list,
            'goal_hat_list': goal_hat_list,
            'scene_names': scene_name_list,
        }

    def _best_goal_loss(self, goal_BK2, gt_goal_B2):
        best_goal_sample_index_B = \
            model_utils.closest_sample_index(goal_BK2,
                                             gt_goal_B2,
                                             metric='goal')
        best_goal_B2 = goal_BK2[torch.arange(goal_BK2.size(0)),
                                best_goal_sample_index_B]
        return (best_goal_B2 - gt_goal_B2).norm(dim=-1).mean()

    def _update_metrics(self,
                        split: Literal['train', 'val', 'test'],
                        best_goal_orig_B2: Tensor,
                        gt_goal_orig_B2: Tensor):

        # Get the methods to compute the metrics.
        fde_method = getattr(self, f'{split}_fde')
        env_col_method = getattr(self, f'{split}_env_collisions')

        # Compute metrics.
        fde_method(best_goal_orig_B2[:, None, :], gt_goal_orig_B2[:, None, :])
        # TODO: env_col_method()

    def _compute_metrics(self, split: Literal['train', 'val', 'test']):
        fde = getattr(self, f'{split}_fde')
        env_col = getattr(self, f'{split}_env_collisions')
        return fde, env_col

    def _log(self,
             split,
             loss,
             fde,
             env_col,
             batch_size):

        # Log metrics and losses.
        log_dict_show = {
            f'{split}_fde': fde,
        }
        self.log_dict(log_dict_show,
                      prog_bar=True,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_size)

        log_dict_hide = {
            f'{split}_loss': loss,
            f'{split}_env_collisions': env_col,
        }
        self.log_dict(log_dict_hide,
                      prog_bar=False,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_size)





    def _shared_step_old(self, batch, batch_idx, phase):
        num_traj = 0
        mse_loss = torch.tensor(0.0, device=self.device)
        for batch_i in zip(*batch):
            (scene_orig_sA2,
             map_mask_1HW,
             transform_matrix,
             dataset_name,
             hom_orig2meters,
             hom_meters2orig,
             hom_meters2mask,
             coord_system) = batch_i

            # Remove batch padding.
            mask = ~(torch.isnan(scene_orig_sA2)).all(dim=(1, 2))
            scene_orig_SA2 = scene_orig_sA2[mask]

            # Project the scene to meters.
            scene_SA2 = utils.project(scene_orig_SA2, hom_orig2meters)

            # Split the scene into observed and predicted trajectories.
            # scene_SA2 is always of length obs_len + pred_len.
            scene_obs_SO2 = scene_SA2[:, :self.obs_len]
            scene_pred_SP2 = scene_SA2[:, -self.pred_len:]
            # Keep a copy of the ground truth in the original coordinates.
            scene_obs_orig_SO2 = scene_orig_SA2[:, :self.obs_len]
            scene_pred_orig_SP2 = scene_orig_SA2[:, -self.pred_len:]

            # Compute goals.
            goals_delta_BK2 = self(scene_obs_SO2)
            if self.current_epoch >= 20:
                print(goals_delta_BK2.norm(dim=-1).max().item())

            goals_BK2 = goals_delta_BK2 + scene_obs_SO2[:, -1].unsqueeze(1)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if self.current_epoch >= 10:
            #     # TODO: debug remove
            #     map_rgb = cv.cvtColor(map_mask_1HW[0].cpu().numpy(), cv.COLOR_GRAY2RGB)
            #     # project the observed, future and goals to the mask
            #     scene_obs_mask_SO2 = utils.project(scene_obs_SO2, hom_meters2mask)
            #     scene_pred_mask_SP2 = utils.project(scene_pred_SP2, hom_meters2mask)
            #     goals_mask_BK2 = utils.project(goals_BK2, hom_meters2mask)
            #     # # draw last observed point
            #     # for i in range(scene_obs_mask_SO2.size(0)):
            #     #     cv.circle(map_rgb, tuple(scene_obs_mask_SO2[i, -1].int().tolist()), 1, (0, 0, 255), -1)
            #     # draw observed points for first pedestrian
            #     for i in range(scene_obs_mask_SO2.size(1)):
            #         cv.circle(map_rgb, tuple(scene_obs_mask_SO2[0, i].int().tolist()), 1, (0, 0, 255), -1)
            #     # draw future points for first pedestrian
            #     for i in range(scene_pred_mask_SP2.size(1)):
            #         cv.circle(map_rgb, tuple(scene_pred_mask_SP2[0, i].int().tolist()), 1, (255, 0, 0), -1)
            #     # draw goals for first pedestrian
            #     for i in range(goals_mask_BK2.size(1)):
            #         cv.circle(map_rgb, tuple(goals_mask_BK2[0, i].int().tolist()), 1, (0, 255, 0), -1)

            #     cv.imshow("observed", map_rgb)
            #     cv.waitKey(0)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            # Compute loss.

            # Targets.
            gt_goals_B2 = scene_pred_SP2[:, -1]

            best_goal_sample_index_B = \
            model_utils.closest_sample_index(goals_BK2,
                                             gt_goals_B2,
                                             metric='goal')

            # Extract the best sample for each pedestrian.
            best_goals_B2 = goals_BK2[torch.arange(goals_BK2.size(0)),
                                      best_goal_sample_index_B]

            # Compute loss.
            # curr_loss = F.mse_loss(best_goals_B2, gt_goals_B2)
            curr_loss = (best_goals_B2 - gt_goals_B2).norm(dim=-1).sum()
            mse_loss += curr_loss

            num_traj += scene_obs_SO2.size(0)

        # Average loss.
        loss = mse_loss / num_traj

        # Log metrics.
        self.log(f'{phase}_loss',
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 batch_size=num_traj)

        return loss









# class GoalLightningCLI(LightningCLI):
#     """Custom LightningCLI that adds some additional functionality."""

#     def add_arguments_to_parser(self, parser):

#         # Link arguments (to avoid duplication).
#         parser.link_arguments("model.init_args.obs_len",
#                               "data.init_args.obs_len")
#         parser.link_arguments("model.init_args.pred_len",
#                               "data.init_args.pred_len")

#         # Debug argument.
#         parser.add_argument(
#             "--debug",
#             action="store_true",
#             help="Enable debug mode."
#         )

#         # Add default logger.
#         parser.set_defaults({
#             "trainer.logger": {
#                 "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
#                 "init_args": {
#                     "save_dir": "logs",
#                     "default_hp_metric": False,
#                 },
#             },
#         })

#         # Add default checkpoint callback.
#         parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
#         parser.set_defaults({"checkpoint_callback.save_top_k": 1,
#                              "checkpoint_callback.monitor": "val_loss",
#                              "checkpoint_callback.mode": "min",
#                              "checkpoint_callback.save_last": True})

#         # Add experiment name argument.
#         parser.add_argument(
#             "--experiment_name",
#             default="unnamed_exp",
#             type=str,
#             help="Name of the experiment"
#         )

#         # Use experiment name in logger.
#         parser.link_arguments("experiment_name",
#                               "trainer.logger.init_args.name")

#         # Checkpoint filename is a function of the experiment name.
#         ckpt_name_fun = \
#             lambda x: f"{x}" + "-{epoch:02d}-{val_loss:.4f}"
#         parser.link_arguments("experiment_name",
#                               "checkpoint_callback.filename",
#                               compute_fn=ckpt_name_fun)

#         # Add default learning rate monitor callback.
#         parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")

#         # Add argument for resuming training.
#         parser.add_argument(
#             "--ckpt_path",
#             type=str,
#             help="Path to the checkpoint to resume training from."
#         )

# def main(args: ArgsType = None):
#     import warnings
#     warnings.filterwarnings("ignore", ".*does not have many workers.*")
#     warnings.filterwarnings("ignore", ".*to avoid having duplicate data.*")

#     # Log git info.
#     sha, diff, branch = model_utils.git_info()
#     logger.info(f"Git - sha={sha} branch={branch} diff='{diff}'")

#     cli = GoalLightningCLI(args=args, run=False)

#     try:
#         cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)
#         val_res = cli.trainer.validate(cli.model,
#                                     cli.datamodule,
#                                     ckpt_path='best')[0]
#         test_res = cli.trainer.test(cli.model,
#                                     cli.datamodule,
#                                     ckpt_path='best')[0]

#         result = {
#             "val_loss": val_res["val_loss"],
#             "test_loss": test_res["test_loss"],
#         }

#         model_utils.save_results(os.path.join(cli.trainer.log_dir,
#                                               "results.yaml"),
#                                  result)
#     except Exception as e:
#         if cli.config.debug:
#             import traceback; traceback.print_exc()
#             import pdb; pdb.post_mortem()
#         else:
#             raise e


# if __name__ == "__main__":
#     main()
