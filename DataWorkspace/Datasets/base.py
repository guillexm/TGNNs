# This is the base class for all datasets.
# We use factory methods to create a GraphDataset instance from given dataset config files.
# These instances can then be fed to model instances.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Any, Dict

# DataConfig contains the configuration for the dataset.
# The code below is a template for the dataset config.
# It will be overridden by the dataset config files for each dataset.
@dataclass
class DataConfig:
    window: int = 12 # past steps given to the model
    horizon: int =12 # forecast length
    root: str = "" # path to the dataset
    split: str = "train" # "train" / "val" / "test"
    #download: bool = True  # auto-download if missing, !!! So far we are not implementing this feature to save time.
    

# 
class GraphDataset(Dataset, ABC):
    """GraphDataset is the base class for all datasets that we will use.
    It inherits from torch.utils.data.Dataset, which is a base class for all PyTorch datasets."""
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.root = cfg.root
        self.split = cfg.split
        self.A: torch.Tensor | None = None     # adjacency matrix
        self.X: torch.Tensor | None = None     # (T, N, F) features of nodes; T: time steps, N: number of nodes, F: number of features per nodes.
        self.y: torch.Tensor | None = None     # (T, N, horizon) target values; T: time steps, N: number of nodes, horizon: forecast length.
        self._prepare()                        # Forces full loading of the dataset in the subclass' constructor.

    # ----- Dataset interface -----
    @abstractmethod
    def __len__(self) -> int: 
        """Return the number of samples in the dataset."""
    @abstractmethod
    def __getitem__(self, idx: int):
        """Return (x_window, A, y_horizon) for sample idx."""

    # ----- subclass hook ---------
    @abstractmethod
    def _prepare(self) -> None:
        """Download/process data; set self.A, self.X, self.y."""

    # ----- handy properties ---------------
    @property
    def num_nodes(self) -> int:
        if self.A is not None:
            return self.A.size(0) #First dimension of the adjacency matrix is the number of nodes.
        return 0

    @property
    def num_features(self) -> int:
        if self.X is not None:
            return self.X.size(-1) #Last dimension of the feature matrix (T, N, F) is the number of features.
        return 0

    """TODO: Add other properties such as time_steps, num_samples, etc."""

    # ----- factory method to create a GraphDataset instance from a given dataset config file ---
    @classmethod
    def from_config(cls, cfg_dict: Dict[str, Any]) -> "GraphDataset":
        return cls(DataConfig(**cfg_dict))