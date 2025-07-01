
#This is the file that deals with the NY-ISO dataset. We inherit from the GraphDataset base class.
from dataworkspace.datasets.base import GraphDataset, DataConfig
import pandas as pd
from typing import Dict, TypedDict
from dataclasses import dataclass

@dataclass
class NYISOConfig(DataConfig):
    #So far no changes with respect to the base class, except that we give base values to window and horizon.
    window: int = 288 #One day
    horizon: int = 12   #One hour

#A list of the NY_ISO zones
ZONES = [
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL",
    "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST", "NYCA"
]

class NYISOFiveMinZonal(GraphDataset):
    """
    5-minute timestep load on the NY-ISO network
    TODO: Add some info on what are good values for the timesteps.
    """
    def __init__(self, cfg: NYISOConfig):
        # the base class calls _prepare()
        super().__init__(cfg)

    def _load_neiso_dataframe(self) -> pd.DataFrame:
        #Load both ny_iso_5_min_2018.csv and ny_iso_5_min_2018.csv
        df2018=pd.read_csv('ny_iso_5_min_2018.csv')
        df2019=pd.read_csv('ny_iso_5_min_2019.csv')
        df_combined = pd.concat([df2018, df2019], ignore_index=True)
        df_combined["Time Stamp"]=pd.to_datetime(df_combined["Time Stamp"])
        df_combined=df_combined.sort_values(by='Time Stamp').reset_index(drop=True)
        
        #Drop empty column
        df_combined = df_combined.loc[:, (df_combined.columns.notnull()) & (df_combined.columns != '')]

        return df_combined
    
    def _prepare(self) -> None:
        #Concatenate the data from the different files if necessary
        #Generate the Panda dataframe from our CSV files
        self.data=self._load_neiso_dataframe()

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
    cfg_dict= NYISOConfig()
    ds = NYISOFiveMinZonal(cfg_dict)

