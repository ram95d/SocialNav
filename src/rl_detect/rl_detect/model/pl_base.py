"""PyTorch Lightning base class for easier configuration through
dependency injection."""

from typing import Optional, Literal, Any
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import EarlyStopping

import rl_detect.model.metrics as metrics
from .metrics import ADE, FDE, Collisions, EnvironmentCollisions
from .social_nce import (
    SocialNceLoss, ISocialNceCompatible, SocialQueryEmbedder, SocialKeyEmbedder
)
from .map_nce import (
    MapNceLoss, IMapNceCompatible, MapQueryEmbedder, MapKeyEmbedder
)
from . import model_utils
from .sampling_info import SamplingInfo


class ConfigurableLitModule(L.LightningModule, ABC):
    """PyTorch Lightning base class for easier configuration through
    dependency injection."""

    def __init__(self,
                 optimizer: Optional[dict] = None,
                 lr_scheduler: Optional[dict] = None,
                 early_stopping: Optional[dict] = None,
                 gradient_clipping: Optional[dict] = None):
        """
            TODO
        """

        super().__init__()

        # Training parameters.
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.early_stopping_cfg = early_stopping
        self.gradient_clipping_cfg = gradient_clipping


    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            return None

        # TODO: maybe validate cfg

        opt_name = self.optimizer_cfg['name']

        optimizer_constructor = getattr(torch.optim, opt_name)

        opt_cfg = self.optimizer_cfg.copy()
        del opt_cfg['name']

        optimizer = optimizer_constructor(self.parameters(), **opt_cfg)

        if self.lr_scheduler_cfg is None \
           or self.lr_scheduler_cfg['name'] is None:
            return optimizer


        lr_scheduler_cfg = self.lr_scheduler_cfg.copy()

        lr_scheduler_name = self.lr_scheduler_cfg['name']
        lr_scheduler_monitor = self.lr_scheduler_cfg.get('monitor', None)
        lr_scheduler_interval = self.lr_scheduler_cfg.get('interval', None)
        del lr_scheduler_cfg['name']
        if lr_scheduler_monitor: del lr_scheduler_cfg['monitor']
        if lr_scheduler_interval: del lr_scheduler_cfg['interval']

        lr_scheduler_constructor = getattr(torch.optim.lr_scheduler,
                                           lr_scheduler_name)


        lr_scheduler = lr_scheduler_constructor(optimizer, **lr_scheduler_cfg)

        if lr_scheduler_monitor is not None:
            pl_scheduler = {
                'scheduler': lr_scheduler,
                'monitor': lr_scheduler_monitor,
                'interval': lr_scheduler_interval
            }
        else:
            pl_scheduler = lr_scheduler

        return [optimizer], [pl_scheduler]

    def configure_callbacks(self):
        callbacks = []

        if self.early_stopping_cfg is not None:
            early_stopping = EarlyStopping(**self.early_stopping_cfg)
            callbacks.append(early_stopping)

        return callbacks

    def configure_gradient_clipping(self,
                                    optimizer,
                                    gradient_clip_val,
                                    gradient_clip_algorithm):
        if self.gradient_clipping_cfg is not None \
           and self.gradient_clipping_cfg['clip']:
            gradient_clip_val = self.gradient_clipping_cfg['value']
            gradient_clip_algorithm = self.gradient_clipping_cfg['algorithm']

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
