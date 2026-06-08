import os
from functools import partial
from multiprocessing import Pool
import json
import pickle
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional
from collections.abc import Callable

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, WeightedRandomSampler
from torch import Tensor
import lightning as L
import pandas as pd
import numpy as np

import model.datasets.traj_utils as traj_utils
import model.datasets.augmentation as augmentation


logger = logging.getLogger(__name__)

class TrajectoryDataset(Dataset):
    def __init__(self,
                 path: str,
                 homographies_path: str,
                 coord_system: Literal['world', 'image'],
                 obs_len: int,
                 pred_len: int,
                 min_obs_len: int,
                 min_pred_len: int,
                 only_visible: bool,
                 fill_missing: bool,
                 max_step_size: float | None,
                 frame_delta: int,
                 cluster_labels_path: str | None,
                 parallel: bool = True,
                 cache_dir: str | None = None):
        super().__init__()

        social_dataset = SocialTrajectoryDataset(
            path=path,
            homographies_path=homographies_path,
            coord_system=coord_system,
            obs_len=obs_len,
            pred_len=pred_len,
            min_obs_len=min_obs_len,
            min_pred_len=min_pred_len,
            only_visible=only_visible,
            fill_missing=fill_missing,
            max_step_size=max_step_size,
            frame_delta=frame_delta,
            parallel=parallel,
            cache_dir=cache_dir
        )

        self.dataset_folder = social_dataset.dataset_folder

        cluster_labels = None
        if cluster_labels_path is None:
            self.weights = torch.ones(len(social_dataset))
        else:
            # Load cluster labels.
            cluster_labels = torch.load(cluster_labels_path)
            # Sampling probabilities for each cluster (inverse of cluster size).
            cluster_probs = 1 / torch.bincount(cluster_labels)
            cluster_probs /= cluster_probs.sum()
            # Sampling weights for each trajectory.
            self.weights = cluster_probs[cluster_labels]

        self.trajs = []
        traj_idx = 0
        for scene, map_mask, dataset_name in social_dataset.scenes:
            for traj in scene:
                label = cluster_labels[traj_idx] if cluster_labels is not None else None
                self.trajs.append((traj, map_mask, dataset_name, label))
                traj_idx += 1

        self.coord_system = coord_system
        self.homographies = social_dataset.homographies

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        """Returns a sample from the dataset."""

        # Get the scene and the map mask path.
        traj, map_mask_path, dataset_name, label = self.trajs[idx]
        homographies = self.homographies[dataset_name]

        # Load the map mask if it exists.
        map_mask = None
        # map_border = None
        if map_mask_path:
            true_map_mask_path = os.path.join(self.dataset_folder,
                                              map_mask_path)

            map_mask = torchvision.io.read_image(true_map_mask_path)

        return (traj,
                map_mask,
                dataset_name,
                homographies,
                self.coord_system)


class SocialTrajectoryDataset(Dataset):
    """A PyTorch Dataset for trajectory data.

    Data is expected to be in the format of a blank-separated file with the
    following columns:
        - frame: The frame number.
        - id: The person identifier.
        - x: The x coordinate of the person.
        - y: The y coordinate of the person.
        - map_path: The path to the map image (optional).

    If path is a folder, all files in the folder will be read and concatenated
    into a single dataset.

    Assumes that the data is labeled at 2.5 fps (0.4 seconds),
    and that the frame id is a multiple of 10.

    The dataset is then split into scenes containing all the visible
    pedestrians in a frame interval of obs_len + pred_len, and each scene is
    returned as a sample.
    A sample has shape: (num_pedestrians, obs_len + pred_len, 2)
    Missing values are set to nan.
    """

    def __init__(self,
                 path: str,
                 homographies_path: str,
                 coord_system: Literal['world', 'image'],
                 obs_len: int,
                 pred_len: int,
                 min_obs_len: int,
                 min_pred_len: int,
                 only_visible: bool,
                 fill_missing: bool,
                 max_step_size: float | None,
                 frame_delta: int,
                 parallel: bool = True,
                 cache_dir: str | None = None):
        """Creates a new TrajectoryDataset.

        Args:
            path: The path to the dataset or a folder containing
                several datasets.
            homographies_path: The path to the homography folder.
            coord_system: The type of coordinates in the dataset.
                Can be 'world' or 'image'. If 'world', the coordinates
                are in the world coordinate system. If 'image', the
                coordinates are in the image coordinate system.
            obs_len: The length of the observed part of the trajectory.
            pred_len: The forecast horizon.
            min_obs_len: The minimum number of observations a pedestrian
                must have in the observed part of the trajectory to be
                included in the dataset.
            min_pred_len: The minimum number of observations a pedestrian
                must have in the predicted part of the trajectory to be
                included in the dataset.
            only_visible: If True, only scenes with all pedestrians visible
                are returned. That is the last observation must not be nan.
            fill_missing: If True, missing observations in the middle
                are filled by interpolating the existing ones.
            max_step_size: The maximum step size between consecutive frames
                for a given pedestrian. Helps to filter out unrealistic
                trajectories.
            parallel: Weather to use a parallel implementation.
        """

        super().__init__()

        # Dataset path.
        self.path = path
        path_is_dir = os.path.isdir(path)

        self.dataset_folder = path if path_is_dir else os.path.dirname(path)

        # List of scenes (tuple (tensor, env_mask_path)).
        self.scenes = []
        self.scene_sizes = []

        # Homographies for each dataset.
        # Dict of dataset_name: str -> (homography_orig2meters: tensor,
        #                               homography_meters2orig: tensor,
        #                               homography_meters2mask: tensor).
        self.homographies = {}

        # Coordinates system.
        self.coord_system = coord_system

        # Initialize the max step size.
        max_step_size = max_step_size if max_step_size is not None else float('inf')

        # Create a cache folder if it does not exist.
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        else:
            HERE = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(HERE, '../../cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Get the list of files that compose the dataset.
        if path_is_dir:
            file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
        else:
            file_list = [path]

        # Compute cache key.
        cache_key = _get_cache_key(file_list,
                                   obs_len,
                                   pred_len,
                                   min_obs_len,
                                   min_pred_len,
                                   only_visible,
                                   fill_missing,
                                   max_step_size)

        # Cache file path for this dataset.
        cache_file = f'{cache_dir}/{cache_key}.pkl'

        # Load the dataset from the cache if it exists.
        if os.path.exists(cache_file):
            logger.info('Loading cached dataset')
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.scenes = data['scenes']
                self.scene_sizes = data['scene_sizes']
                self.homographies = data['homographies']
                self.coord_system = data['coord_system']

            # Convert the scenes to tensors.
            self._scenes_to_tensor()
            return

        logger.info('Building dataset')

        for filename in file_list:
            self.scenes.extend(_process_file(filename=filename,
                                             obs_len=obs_len,
                                             pred_len=pred_len,
                                             min_obs_len=min_obs_len,
                                             min_pred_len=min_pred_len,
                                             only_visible=only_visible,
                                             fill_missing=fill_missing,
                                             max_step_size=max_step_size,
                                             frame_delta=frame_delta,
                                             parallel=parallel))

        self.scene_sizes = [scene.shape[0] for scene, _, _ in self.scenes]
        self.scene_sizes = torch.tensor(self.scene_sizes, dtype=torch.int64)

        # Iterate over unique dataset names to load the homographies.
        for _, _, dataset_name in self.scenes:
            if dataset_name not in self.homographies:
                homographies = _load_homographies(dataset_name, homographies_path)
                self.homographies[dataset_name] = homographies

        # Save the scenes to a file.
        logger.info('Saving dataset to cache')
        with open(cache_file, 'wb') as f:
            data = {
                'scenes': self.scenes,
                'scene_sizes': self.scene_sizes,
                'homographies': self.homographies,
                'coord_system': self.coord_system,
            }
            pickle.dump(data, f)

        # Convert the scenes to tensors.
        self._scenes_to_tensor()

    def _scenes_to_tensor(self):
        """Converts the scenes to a tensor."""

        # Convert the scenes to pytorch tensors.
        self.scenes = \
            [(torch.from_numpy(scene), map_mask, dataset_name)
             for scene, map_mask, dataset_name in self.scenes]

    def __len__(self):
        """Returns the number of samples in the dataset."""

        return len(self.scenes)

    def __getitem__(self, idx):
        """Returns a sample from the dataset."""

        # Get the scene and the map mask path.
        scene, map_mask_path, dataset_name = self.scenes[idx]
        homographies = self.homographies[dataset_name]

        # Load the map mask if it exists.
        map_mask = None
        # map_border = None
        if map_mask_path:
            true_map_mask_path = os.path.join(self.dataset_folder,
                                              map_mask_path)

            map_mask = torchvision.io.read_image(true_map_mask_path)

        return (scene,
                map_mask,
                dataset_name,
                homographies,
                self.coord_system)


def _process_file(filename: str,
                  obs_len: int,
                  pred_len: int,
                  min_obs_len: int,
                  min_pred_len: int,
                  only_visible: bool,
                  fill_missing: bool,
                  max_step_size: float,
                  frame_delta: int,
                  parallel: bool
                  ) -> list[tuple[np.ndarray,
                                  Optional[str],
                                  Optional[str]]]:
    """Processes a file and returns the scenes.

    Args:
        filename: Filename of the file to process.
        obs_len: The length of the observed part of the trajectory.
        pred_len: The forecast horizon.
        columns: The columns of the dataset.
        min_obs_len: The minimum number of observations a pedestrian
            must have in the observed part of the trajectory to be
            included in the dataset.
        min_pred_len: The minimum number of observations a pedestrian
            must have in the predicted part of the trajectory to be
            included in the dataset.
        only_visible: If True, only scenes with all pedestrians visible
            are returned. That is the last observation must not be nan.
        fill_missing: If True, missing observations in the middle
            are filled by interpolating the existing ones.
        max_step_size: The maximum step size between consecutive frames
            for a given pedestrian. Helps to filter out unrealistic
            trajectories.
        parallel: Weather to use a parallel implementation.

    Returns:
        A list of scenes.
    """

    # Expected dataset columns.
    COLUMNS = ['frame', 'id', 'x', 'y', 'map_path']

    raw_dataset = pd.read_csv(filename,
                              sep=r'\s+',
                              header=None,
                              names=COLUMNS)

    has_map_mask = not raw_dataset['map_path'].isna().any() # type: ignore

    frames = raw_dataset['frame'].unique().astype(int)

    # Iterate over all windows of frames.
    frame_window_size = obs_len + pred_len
    frame_windows = []
    for i in range(len(frames) - frame_window_size + 1):
        start_frame = frames[i].item()
        end_frame = frames[i + frame_window_size - 1]
        if end_frame - start_frame == (frame_window_size - 1) * frame_delta:
            frame_windows.append((start_frame, end_frame))

    process_frame_window_partial = partial(_process_frame_window,
                                           frame_delta=frame_delta,
                                           obs_len=obs_len,
                                           pred_len=pred_len,
                                           raw_dataset=raw_dataset,
                                           min_obs_len=min_obs_len,
                                           min_pred_len=min_pred_len,
                                           fill_missing=fill_missing,
                                           only_visible=only_visible,
                                           max_step_size=max_step_size,
                                           has_map_mask=has_map_mask)

    if parallel:
        scenes = []

        # Process the frame windows in parallel.
        with Pool(processes=None) as p:
            scenes.extend(p.map(process_frame_window_partial, frame_windows))

    else:
        # Process the frame windows sequentially.
        scenes = [ process_frame_window_partial(frame_window)
                   for frame_window in frame_windows ]

    # Filter out the None values (scenes with no pedestrians).
    scenes = [scene for scene in scenes if scene is not None]

    return scenes

def _process_frame_window(frame_window: tuple[int, int],
                          frame_delta: int,
                          obs_len: int,
                          pred_len: int,
                          raw_dataset: pd.DataFrame,
                          min_obs_len: int,
                          min_pred_len: int,
                          fill_missing: bool,
                          only_visible: bool,
                          max_step_size: float,
                          has_map_mask: bool
                          ) -> Optional[tuple[np.ndarray,
                                              Optional[str],
                                              Optional[str]]]:
    """Processes a window of frames and returns the corresponding scene.

    Args:
        frame_window: Tuple containing the start and end frame of the window.
        frame_delta: The difference between consecutive frame IDs.
        obs_len: The length of the observed part of the trajectory.
        pred_len: The forecast horizon.
        raw_dataset: The raw dataset.
        min_obs_len: The minimum number of observations a pedestrian
            must have in the observed part of the trajectory to be
            included in the dataset.
        min_pred_len: The minimum number of observations a pedestrian
            must have in the predicted part of the trajectory to be
            included in the dataset.
        fill_missing: If True, missing observations in the middle
            are filled by interpolating the existing ones.
        only_visible: If True, only scenes with all pedestrians visible
            are returned. That is the last observation must not be nan.
        max_step_size: The maximum step size between consecutive frames
            for a given pedestrian. Helps to filter out unrealistic
            trajectories.
        has_map_mask: Whether the dataset has the map mask column.

    Returns:
        The scene if it contains at least one pedestrian, otherwise None.
    """

    # Extract the observations in the window.
    start_frame, end_frame = frame_window
    window_df = raw_dataset[
        (raw_dataset['frame'] >= start_frame) &
        (raw_dataset['frame'] <= end_frame)
    ]
    # IDs of the pedestrians in the window.
    pedestrians = window_df['id'].unique() # type: ignore
    num_pedestrians = len(pedestrians)

    # Create an empty scene, to be filled with the pedestrians' positions.
    scene = torch.full((num_pedestrians, obs_len + pred_len, 2), np.nan)

    for frame in range(start_frame, end_frame + 1, frame_delta):
        # Get the time index in the scene.
        time = int((frame - start_frame) / frame_delta)
        # Get the dataframe for the current frame.
        frame_df = window_df[window_df['frame'] == frame]
        # Iterate over all pedestrians in the window
        # (not just the ones in the current frame).
        for i, pedestrian in enumerate(pedestrians):
            # Get the pedestrian's position in the frame.
            position = frame_df[frame_df['id'] == pedestrian][['x', 'y']]
            # If the pedestrian is in the frame, update the scene.
            if len(position) > 0:
                # Insert the pedestrian's position in the scene.
                scene[i, time] = torch.tensor(position.values[0]) # type: ignore

    if fill_missing:
        # Fill the missing observations by interpolating the existing ones.
        traj_utils.fill_missing(scene)

    scene = \
        traj_utils.drop_if_not_enough_observations(scene, obs_len, min_obs_len)
    scene = \
        traj_utils.drop_if_not_enough_predictions(scene, pred_len, min_pred_len)
    scene = \
        traj_utils.drop_if_unrealistic_step_size(scene, max_step_size)

    if only_visible:
        # Check if all the pedestrians are visible in the scene.
        # A pedestrian is visible if the last observation
        # is not nan.
        scene = traj_utils.keep_only_visible(scene, obs_len)

    # Check that the scene has at least one pedestrian.
    if scene.shape[0] == 0:
        return None

    map_mask_path = None
    dataset_name = None
    if has_map_mask:
        # Load the map mask path.
        # Use the map mask from the last observed frame
        # (even though for now it is the same for all frames).
        last_obs_frame_id = start_frame + obs_len * frame_delta
        frame_df = window_df[window_df['frame'] == last_obs_frame_id]
        map_mask_path = frame_df['map_path'].values[0] # type: ignore

        # TODO: Remove this hack: don't want to use dataset name in the future.
        # Dataset name.
        base_name = os.path.basename(map_mask_path)
        # Remove '-mask.png' from the name.
        dataset_name = os.path.splitext(base_name)[0][:-5]

    return scene.numpy(), map_mask_path, dataset_name


def _get_cache_key(file_list: list[str],
                   obs_len: int,
                   pred_len: int,
                   min_obs_len: int,
                   min_pred_len: int,
                   only_visible: bool,
                   fill_missing: bool,
                   max_step_size: float) -> str:
    """Computes the cache key for the dataset.

    The cache key is computed using:
    - The hash of the content of the files.
    - Configuration parameters:
        - obs_len
        - pred_len
        - min_obs_len
        - min_pred_len
        - only_visible
        - fill_missing
        - max_step_size

    Note: current cache key doesn't account for changes in homography files.

    Args:
        file_list: The list of files in the dataset.

    Returns:
        The cache key.
    """

    # Sort the file list to ensure the same key is computed.
    file_list = sorted(file_list)

    # Compute hash of the content of the files.
    hash = hashlib.sha256()
    for filename in file_list:
        with open(filename, 'rb') as f:
            hash.update(f.read())
    hash = hash.hexdigest()

    # Compute the cache key.
    key = f'{hash}' \
          f'_{obs_len}' \
          f'_{pred_len}' \
          f'_{min_obs_len}' \
          f'_{min_pred_len}' \
          f'_{only_visible}' \
          f'_{fill_missing}' \
          f'_{max_step_size}'

    return key


def _load_homographies(scene_name: str,
                       homographies_folder: str) -> dict[str, Tensor]:
    """Retrieves the homography matrix (json file).

    Args:
        scene_name: The name of the scene.
        dataset_folder: The folder containing the dataset.
            Expected to contain the 'meters' and 'mask' folders,
            each containing the homography matrices for the scene
            in JSON format.

    Returns:
        Tuple containing 3 homography matrices:
        - homography_orig2meters: The homography matrix from the original
            coordinate system to the meters coordinate system.
        - homography_meters2orig: The homography matrix from the meters
            coordinate system to the original coordinate system.
        - homography_meters2mask: The homography matrix from the meters
            coordinate system to the mask coordinate system.
    """

    homography_orig2meters_path = \
        os.path.join(homographies_folder, 'meters', f'{scene_name}.json')

    # Load the homography_orig2meters from the JSON file.
    with open(homography_orig2meters_path, 'r') as f:
        homography_orig2meters = torch.tensor(json.load(f), dtype=torch.float32)

    # Compute the inverse of homography_orig2meters.
    homography_meters2orig = torch.inverse(homography_orig2meters)

    # Load the homography_meters2mask from the JSON file.
    homography_meters2mask_path = \
        os.path.join(homographies_folder, 'mask', f'{scene_name}.json')
    with open(homography_meters2mask_path, 'r') as f:
        homography_meters2mask = torch.tensor(json.load(f), dtype=torch.float32)

    # Compute the inverse of homography_meters2mask.
    homography_mask2meters = torch.inverse(homography_meters2mask)

    return {
        'orig2meters': homography_orig2meters,
        'meters2orig': homography_meters2orig,
        'meters2mask': homography_meters2mask,
        'mask2meters': homography_mask2meters,
    }


class SocialTrajectorySubset(Dataset):
    """Custom subset class for TrajectoryDataset that preserves scene
    size information, allowing to be used in conjunction with the
    TrajBatchSampler.
    """

    def __init__(self, dataset: SocialTrajectoryDataset, indices: torch.Tensor):
        """
        Args:
            dataset: The TrajectoryDataset to subset
            indices: List of indices to select from the dataset
        """
        self.dataset = dataset
        self.indices = indices
        # Preserve scene sizes for the selected indices.
        self.scene_sizes = dataset.scene_sizes[indices]

    def __getitem__(self, idx):
        """Get an item from the subset."""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """Return the length of the subset."""
        return len(self.indices)

class TrajectorySubset(Dataset):
    """Custom subset class for TrajectoryDataset that preserves the
    weights of the trajectories, allowing to be used in conjunction
    with the WeightedRandomSampler.
    """

    def __init__(self, dataset: TrajectoryDataset, indices: torch.Tensor):
        """
        Args:
            dataset: The TrajectoryDataset to subset
            indices: List of indices to select from the dataset
        """

        self.dataset = dataset
        self.indices = indices
        # Preserve weights for the selected indices.
        self.weights = dataset.weights[indices]

    def __getitem__(self, idx):
        """Get an item from the subset."""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """Return the length of the subset."""
        return len(self.indices)


class BaseTrajectoryDataModule(L.LightningDataModule, ABC):
    """Base DataModule for trajectory data with common functionality."""

    def __init__(self,
                 path: str,
                 homographies_path: str,
                 coord_system: Literal['world', 'image'],
                 obs_len: int,
                 pred_len: int,
                 min_obs_len: int,
                 min_pred_len: int,
                 only_visible: bool,
                 fill_missing: bool,
                 max_step_size: float | None,
                 frame_delta: int,
                 center_scene: bool,
                 rotate_prob: float,
                 flip_prob: float,
                 noise_prob: float,
                 noise_scale: float,
                 batch_size: int,
                 num_workers: int = 0):
        super().__init__()

        self.path = path
        self.homographies_path = homographies_path
        self.coord_system = coord_system
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.only_visible = only_visible
        self.fill_missing = fill_missing
        self.min_obs_len = min_obs_len
        self.min_pred_len = min_pred_len
        self.max_step_size = max_step_size
        self.frame_delta = frame_delta
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Augmentation transforms.
        center_ref_observation = obs_len - 1 if center_scene else None
        self.transforms_train = \
            self._create_augmentation(center_ref_observation,
                                      rotate_prob,
                                      flip_prob,
                                      noise_prob,
                                      noise_scale)
        self.transforms_val = \
            self._create_augmentation(center_ref_observation, 0, 0, 0, 0)
        self.transforms_test = \
            self._create_augmentation(center_ref_observation, 0, 0, 0, 0)

        # Datasets (lazy initialization).
        self.dataset: Dataset | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _create_augmentation(self, center_ref, rotate_p, flip_p, noise_p, noise_s):
        return augmentation.TrajectoryAugmentation(
            center_ref_observation=center_ref,
            shift_prob=0,
            rotate_prob=rotate_p,
            flip_prob=flip_p,
            noise_prob=noise_p,
            noise_scale=noise_s
        )

    @property
    def dataset_class(self) -> type[Dataset]:
        raise NotImplementedError

    @property
    def subset_class(self) -> type[Dataset]:
        raise NotImplementedError

    @property
    def collate_fn(self) -> Callable[[list], tuple]:
        raise NotImplementedError

    def get_dataset_kwargs(self, split=None):
        kwargs = {
            'homographies_path': self.homographies_path,
            'coord_system': self.coord_system,
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'min_obs_len': self.min_obs_len,
            'min_pred_len': self.min_pred_len,
            'only_visible': self.only_visible,
            'fill_missing': self.fill_missing,
            'frame_delta': self.frame_delta,
        }
        if split == 'test':
            kwargs['max_step_size'] = None
        else:
            kwargs['max_step_size'] = self.max_step_size
        return kwargs

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        already_split = (
            os.path.isdir(self.path) and
            os.path.exists(os.path.join(self.path, 'train')) and
            os.path.exists(os.path.join(self.path, 'val')) and
            os.path.exists(os.path.join(self.path, 'test'))
        )
        if not already_split:
            self._setup_not_split()
        else:
            self._setup_already_split(stage)

    def _setup_already_split(self, stage: str):
        if stage == 'fit':
            self._setup_train()
            self._setup_val()
        elif stage == 'validate':
            self._setup_val()
        elif stage == 'test':
            self._setup_test()

    def _setup_not_split(self):
        if self.dataset is not None:
            return

        self.dataset = self.dataset_class(path=self.path, **self.get_dataset_kwargs())
        dataset_size = len(self.dataset)
        indices = torch.arange(dataset_size)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)

        self.train_dataset = self._create_subset(indices[:train_size], 'train')
        self.val_dataset = self._create_subset(indices[train_size:train_size+val_size], 'val')
        self.test_dataset = self._create_subset(indices[train_size+val_size:], 'test')

    def _create_subset(self, indices, split):
        SubsetClass = self.subset_class if split == 'train' else Subset
        return SubsetClass(self.dataset, indices.tolist() if SubsetClass == Subset else indices)

    def _setup_train(self):
        if self.train_dataset is None:
            self.train_dataset = self.dataset_class(
                path=os.path.join(self.path, 'train'),
                **self.get_dataset_kwargs('train')
            )

    def _setup_val(self):
        if self.val_dataset is None:
            self.val_dataset = self.dataset_class(
                path=os.path.join(self.path, 'val'),
                **self.get_dataset_kwargs('val')
            )

    def _setup_test(self):
        if self.test_dataset is None:
            self.test_dataset = self.dataset_class(
                path=os.path.join(self.path, 'test'),
                **self.get_dataset_kwargs('test')
            )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, training=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

    @abstractmethod
    def _create_dataloader(self, dataset, training=False) -> DataLoader:
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        scene_bSA2, map_mask_B1HW, *rest = batch
        if self.trainer.training:
            transform = self.transforms_train
        elif self.trainer.validating:
            transform = self.transforms_val
        elif self.trainer.testing:
            transform = self.transforms_test
        else:
            transform = None

        if transform:
            scene_bSA2, _, inv_transform_matrix_b33 = transform(scene_bSA2)
        else:
            inv_transform_matrix_b33 = torch.eye(3, device=scene_bSA2.device).\
                                             unsqueeze(0).\
                                             repeat(scene_bSA2.size(0), 1, 1)

        return scene_bSA2, map_mask_B1HW, inv_transform_matrix_b33, *rest


class TrajectoryDataModule(BaseTrajectoryDataModule):
    """DataModule for individual trajectories with cluster labels support."""

    def __init__(self, cluster_labels_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cluster_labels_path = cluster_labels_path

    @property
    def dataset_class(self):
        return TrajectoryDataset

    @property
    def subset_class(self):
        return TrajectorySubset

    @property
    def collate_fn(self):
        return collate_fn

    def get_dataset_kwargs(self, split=None):
        kwargs = super().get_dataset_kwargs(split)
        kwargs['cluster_labels_path'] = self.cluster_labels_path if split == 'train' else None
        return kwargs

    def _create_dataloader(self, dataset, training=False):
        if training:
            sampler = WeightedRandomSampler(dataset.weights,
                                            len(dataset),
                                            replacement=True)
            return DataLoader(dataset,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers,
                              persistent_workers=True,
                              collate_fn=self.collate_fn)
        else:
            return DataLoader(dataset,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              persistent_workers=True,
                              collate_fn=self.collate_fn)

class SocialTrajectoryDataModule(BaseTrajectoryDataModule):
    """DataModule for social trajectories with batch sampling support."""

    @property
    def dataset_class(self):
        return SocialTrajectoryDataset

    @property
    def subset_class(self):
        return SocialTrajectorySubset

    @property
    def collate_fn(self):
        return social_collate_fn

    def _create_dataloader(self, dataset, training=False):
        if training:
            batch_sampler = TrajBatchSampler(dataset,
                                             self.batch_size,
                                             shuffle=training,
                                             drop_last=True)
            return DataLoader(dataset,
                              batch_sampler=batch_sampler,
                              num_workers=self.num_workers,
                              persistent_workers=True,
                              collate_fn=self.collate_fn)
        else:
            # For validation and test, use standard batching
            return DataLoader(dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              persistent_workers=True,
                              collate_fn=self.collate_fn)


class TrajBatchSampler(Sampler):
    """Samples batched elements by yielding a mini-batch of indices.

    Tries to generate mini-batches with approximately equal number of
    pedestrians in each batch. Incude a number of sequences in each batch
    such that the total number of pedestrians in each batch is equal to
    or greater than the batch size.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        generator (Generator): Generator used in sampling.
    """

    def __init__(self,
                 data_source: SocialTrajectoryDataset,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.scene_sizes)

        if self.shuffle:
            indices = torch.randperm(len(self.data_source))
        else:
            indices = torch.arange(len(self.data_source))
        scene_sizes = self.data_source.scene_sizes[indices]

        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, scene_sizes):
            batch.append(idx.item())
            total_num_peds += num_peds
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Approximated number of batches.
        # The order of trajectories can be shuffled, so this number can vary from run to run.
        # if self.drop_last:
        #     return sum(self.data_source.scene_sizes) // self.batch_size
        # else:
        #     return (sum(self.data_source.scene_sizes) + self.batch_size - 1) // self.batch_size
        return None


def collate_fn(data: list[tuple[Tensor,
                                Tensor,
                                str,
                                dict[str, Tensor],
                                str]]
               ) -> tuple[Tensor,
                          Tensor,
                          list[str],
                          dict[str, Tensor],
                          list[str]]:
    """Collate function for the DataLoader.

    Since the scenes have different number of pedestrians, the scenes are
    padded to the same length, using nan as padding value.
    So that the scenes can be stacked into a single batch tensor.

    Args:
        scenes: A list of scene and map mask pairs.
            Shape: (num_pedestrians, obs_len + pred_len, 2).

    Returns:
        The batch tensor.
        Shape: (batch_size, max_num_pedestrians, obs_len + pred_len, 2).
    """

    (trajs,
     map_masks,
     dataset_names,
     homographies,
     coord_system_list) = zip(*data)

    # Stack the trajectories along a new batch dimension.
    traj_B1T2 = torch.stack(trajs, dim=0)[:, None, :, :]

    # Find maximum dimensions across all map masks
    max_h = max(mask.shape[1] for mask in map_masks)
    max_w = max(mask.shape[2] for mask in map_masks)

    # Pad and stack map masks
    padded_masks = []
    for mask in map_masks:
        h_pad = max_h - mask.shape[1]
        w_pad = max_w - mask.shape[2]
        padded_mask = torch.nn.functional.pad(mask, (0, w_pad, 0, h_pad), value=0)
        padded_masks.append(padded_mask)

    map_masks_B1HW = torch.stack(padded_masks, dim=0)

    return (traj_B1T2,
            map_masks_B1HW,
            dataset_names,
            homographies,
            coord_system_list)

def social_collate_fn(data: list[tuple[Tensor,
                                       Tensor,
                                       str,
                                       dict[str, Tensor],
                                       str]]
                     ) -> tuple[Tensor,
                                Tensor,
                                list[str],
                                dict[str, Tensor],
                                list[str]]:
    """Collate function for the DataLoader.

    Since the scenes have different number of pedestrians, the scenes are
    padded to the same length, using nan as padding value.
    So that the scenes can be stacked into a single batch tensor.

    Args:
        scenes: A list of scene and map mask pairs.
            Shape: (num_pedestrians, obs_len + pred_len, 2).

    Returns:
        The batch tensor.
        Shape: (batch_size, max_num_pedestrians, obs_len + pred_len, 2).
    """

    (scenes,
     map_masks,
     dataset_names,
     homographies,
     coord_system_list) = zip(*data)

    # Stack the scenes along a new batch dimension.
    scenes_batch_BSA2 = torch.nn.utils.rnn.pad_sequence(scenes,
                                                        batch_first=True,
                                                        padding_value=np.nan)

    # Find maximum dimensions across all map masks
    max_h = max(mask.shape[1] for mask in map_masks)
    max_w = max(mask.shape[2] for mask in map_masks)

    # Pad and stack map masks
    padded_masks = []
    for mask in map_masks:
        h_pad = max_h - mask.shape[1]
        w_pad = max_w - mask.shape[2]
        padded_mask = torch.nn.functional.pad(mask, (0, w_pad, 0, h_pad), value=0)
        padded_masks.append(padded_mask)

    map_masks_B1HW = torch.stack(padded_masks, dim=0)

    return (scenes_batch_BSA2,
            map_masks_B1HW,
            dataset_names,
            homographies,
            coord_system_list)
