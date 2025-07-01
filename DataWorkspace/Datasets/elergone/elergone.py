from dataworkspace.datasets.base import GraphDataset, DataConfig
import dataworkspace.paths as paths
from dataclasses import dataclass
import torch
import tsl
import pandas as pd
from pathlib import Path



ROOT=paths.dataset_root("elergone")
class ElergoneConfig(DataConfig):
    root : str = str(ROOT)

class ElergoneDataset(GraphDataset):
    """Wrapper arround the tsl.Elergone dataset to make it inherit from our GraphDataset base class"""
    def __init__(self, cfg: ElergoneConfig):
        self.root=ElergoneConfig.root
        # the base class calls _prepare()
        super().__init__(cfg)
    
    def _prepare(self) -> None:
        #Call the constructor from Elergone
        eler=tsl.datasets.Elergone(root=self.cfg.root)
        self.df=eler.load()[0]

    #BOILERPLATE:
    def __len__(self) -> int:
        # 1 sample (until sliding-window logic is added)
        return 1

    def __getitem__(self, idx: int):
        assert self.X is not None
        assert self.y is not None
        return self.X[0], self.A, self.y[0]

    
if __name__ == "__main__":
    
    # basic boilerplate config
    cfg_dict= ElergoneConfig()
    df = ElergoneDataset(cfg_dict).dataframe()
    df.to_csv("elergone.csv")