from abc import ABC, abstractmethod

from .renderer import NeRFRenderer


class BaseNeRFNetwork(ABC, NeRFRenderer):
    @abstractmethod
    def __init__(
        self,
        encoding: str,
        encoding_dir: str,
        num_layers: int,
        hidden_dim: int,
        geo_feat_dim: int,
        num_layers_color: int,
        hidden_dim_color: int,
        bound: int,
        **kwargs,
    ):
        return super().__init__(bound, **kwargs)

    @abstractmethod
    def forward(self, x, d):
        raise NotImplementedError

    @abstractmethod
    def density(self, x):
        raise NotImplementedError

    @abstractmethod
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        raise NotImplementedError
