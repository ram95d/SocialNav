"""Dataset for generating random mask images.

Useful to train a mask encoder model, through a simple autoencoder setup.
"""

import logging

import torch
from torch.utils.data import IterableDataset, DataLoader
import lightning as L
import numpy as np
import cv2 as cv


logger = logging.getLogger(__name__)


class MaskRandomDataset(IterableDataset):
    def __init__(self, width: int, height: int, seed: int, size: int):
        """Build a dataset that generates random mask images.

        Args:
            width: Width of the mask image.
            height: Height of the mask image.
            seed: Seed for dataset reproducibility.
            size: Maximum number of mask images to generate.
                Set to -1 for infinite generation.
        """

        super().__init__()

        self.width = width
        self.height = height
        self.size = size

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __iter__(self):
        count = 0
        limit_reached = False
        while not limit_reached:
            mask = self._gen_mask_img(self.width, self.height)
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            yield mask

            if self.size > 0:
                count += 1
                limit_reached = count == self.size

    def __len__(self):
        # For epoch size.
        return self.size if self.size > 0 else 1000

    @staticmethod
    def _gen_mask_img(width: int, height: int) -> np.ndarray:
        """Generates a mask image with random shapes.

        Args:
            width: Width of the mask image.
            height: Height of the mask image.

        Returns:
            A mask image with random shapes.
        """

        mask = np.ones((height, width), dtype=np.uint8) * 255

        num_shapes = np.random.randint(1, 5)
        for _ in range(num_shapes):
            shape = np.random.choice(['rectangle', 'circle', 'triangle'])
            if shape == 'rectangle':
                x = np.random.randint(0, width - 1)
                y = np.random.randint(0, height - 1)
                w = np.random.randint(1, width - x)
                h = np.random.randint(1, height - y)
                cv.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
            elif shape == 'circle':
                x = np.random.randint(0, width - 1)
                y = np.random.randint(0, height - 1)
                r = np.random.randint(1, min(width - x, height - y))
                cv.circle(mask, (x, y), r, 0, -1)
            elif shape == 'triangle':
                x1 = np.random.randint(0, width)
                y1 = np.random.randint(0, height)
                x2 = np.random.randint(0, width)
                y2 = np.random.randint(0, height)
                x3 = np.random.randint(0, width)
                y3 = np.random.randint(0, height)
                triangle_cnt = np.array([(x1, y1), (x2, y2), (x3, y3)])
                cv.drawContours(mask, [triangle_cnt], 0, 0, -1)

        return mask


class MaskDataModule(L.LightningDataModule):
    def __init__(self,
                 width: int,
                 height: int,
                 train_seed: int,
                 val_seed: int,
                 val_size: int,
                 test_seed: int,
                 test_size: int,
                 batch_size: int):
        super().__init__()

        self.width = width
        self.height = height

        self.train_seed = train_seed

        self.val_size = val_size
        self.val_seed = val_seed

        self.test_size = test_size
        self.test_seed = test_seed

        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        dataset = MaskRandomDataset(self.width,
                                    self.height,
                                    self.train_seed,
                                    -1)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=0)

    def val_dataloader(self):
        dataset = MaskRandomDataset(self.width,
                                    self.height,
                                    self.val_seed,
                                    self.val_size)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=0)

    def test_dataloader(self):
        dataset = MaskRandomDataset(self.width,
                                    self.height,
                                    self.test_seed,
                                    self.test_size)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=0)
