from dataworkspace.datasets.base import GraphDataset, DataConfig
import dataworkspace.paths as paths
from typing import Literal, Dict, Type
from dataclasses import dataclass
from tsl.datasets import PeMS03, PeMS04, PeMS07, PeMS08, MetrLA, PemsBay

# Dataset name-class mapping:

_DATASET_REGISTRY= {
    "metrla": MetrLA,
    "pemsbay": PemsBay,
    "pems03": PeMS03,
    "pems04": PeMS04,
    "pems07": PeMS07,
    "pems08": PeMS08,
}

ROOT=paths.dataset_root("torch_traffic")

@dataclass
class TorchTrafficConfig(DataConfig):
    set : Literal["pems03", "pems04", "pems07", "pems08", "metrla", "pemsbay"] = "pems03" # Decide which dataset to use
    root : str = str(ROOT)
    
class TorchTrafficDataset(GraphDataset):
    """Wrapper arround the tsl.Elergone dataset to make it inherit from our GraphDataset base class"""
    def __init__(self, cfg: TorchTrafficConfig):
        self.root=TorchTrafficConfig.root
        self.set=TorchTrafficConfig.set
        # the base class calls _prepare()
        super().__init__(cfg)
    
    def _prepare(self) -> None:
        ds_cls = _DATASET_REGISTRY[self.set]
        self.df=ds_cls.load()[0]

    #BOILERPLATE:
    def __len__(self) -> int:
        # 1 sample (until sliding-window logic is added)
        return 1

    def __getitem__(self, idx: int):
        assert self.X is not None
        assert self.y is not None
        return self.X[0], self.A, self.y[0]

if __name__ == "__main__":
    cfg_dict= TorchTrafficConfig(set="metrla")
    df = TorchTrafficDataset(cfg_dict).dataframe()
    df.to_csv("torch_traffic_metrla.csv")