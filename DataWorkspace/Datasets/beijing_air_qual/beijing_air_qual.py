from dataworkspace.datasets.base import GraphDataset, DataConfig
import dataworkspace.paths as paths
from typing import Literal
import tsl

ROOT=paths.dataset_root("beijing_air_qual")

class BAQConfig(DataConfig):
    root : str = str(ROOT)
    option : Literal["big", "small"] = "small" # Decide if we use the small dataset (only Beijing) or the full dataset (43 chinese cities)

class BAQDataset(GraphDataset):
    """Wrapper arround the tsl.Elergone dataset to make it inherit from our GraphDataset base class"""
    def __init__(self, cfg: BAQConfig):
        self.root=BAQConfig.root
        self.option=BAQConfig.option
        # the base class calls _prepare()
        super().__init__(cfg)
    
    def _prepare(self) -> None:
        #Call the constructor from Elergone
        if(self.option=="big"):
            air_qual=tsl.datasets.AirQuality(root=self.cfg.root)
        else:
            air_qual=tsl.datasets.AirQuality(root=self.cfg.root, small=True)

        self.df=air_qual.load()[0]

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
    cfg_dict= BAQConfig()
    df = BAQDataset(cfg_dict).dataframe()
    df.to_csv("beijing_air_quality.csv")


