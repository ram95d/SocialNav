import dataclasses
from typing import Literal

@dataclasses.dataclass
class SamplingInfo:
    """Information about the sampling capabilities of the model."""

    noise_dim: int
    noise_distrib: Literal['gaussian', 'uniform']
