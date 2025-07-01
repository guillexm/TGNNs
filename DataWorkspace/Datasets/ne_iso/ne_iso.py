#This is the file that deals with the NE-ISO dataset. We inherit from the GraphDataset base class.
import os
from dataworkspace.datasets.base import GraphDataset, DataConfig
from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict
import torch
import pandas as pd
from collections import defaultdict
import numpy as np


@dataclass
class NEISOConfig(DataConfig):
    start_year: int = 2019 # Can be between 2019 and 2024, needs to be less or equal to end_year
    end_year: int = 2024 # Can be between 2019 and 2024, needs to be higher or equal to start_year
    demand_source: Literal["rt", "da", "both"] = "rt" # Dataset has both hourly real time demand and day ahead prediction. This decides which one we use.
    use_weather: bool = True # Decide if we want to use weather data as a feature.
    time_encoding: Literal["cyclic", "none"] = "cyclic" # Decide the format for our time features.



ZONES = [
    "ME", "NH", "VT", "CT", "RI",
    "SEMA", "WCMA", "NEMA",
]

class NEISOHourlyZonal(GraphDataset):
    """Hourly demand & weather for the eight ISO-NE zones (SMD Hourly files).
    Selecting the whole 5 years of data gives us 43824 data points."""
    def __init__(self, cfg: NEISOConfig):
        #Start year and end year check:
        if not (2018 < cfg.start_year <= cfg.end_year < 2024):
            raise ValueError(
                f"Invalid years: start_year={cfg.start_year}, "
                f"end_year={cfg.end_year}. "
                "Must satisfy 2018 < start_year <= end_year < 2024."
            )
        self.start_year=cfg.start_year
        self.end_year=cfg.end_year
        self.demand_source=cfg.demand_source
        self.use_weather=cfg.use_weather
        self.time_encoding=cfg.time_encoding
        # the base class calls _prepare()
        super().__init__(cfg)

    
    def _load_neiso_dataframe(self) -> dict[str, pd.DataFrame]:
        """From our config, loads the relevant datapoints in a panda DataFrame dictionary, where the indices are the different zones"""
        # we'll accumulate a list of yearly DataFrames per zone
        zone_data_list: dict[str, list[pd.DataFrame]] = defaultdict(list)
        #Decide which files to load based on start year and end year
        for year in range(self.start_year, self.end_year + 1):
            fp = os.path.join(self.root, f"{year}_smd_hourly.xlsx")
            for zone in ZONES:
                df_zone = self._load_zone_year(fp, zone)
                zone_data_list[zone].append(df_zone)

        #We now concatenate everything into one df for each zone:
        zone_data = {
            zone: pd.concat(dfs, ignore_index=True)
                     .sort_values(["Date", "Hr_End"])
                     .reset_index(drop=True)
            for zone, dfs in zone_data_list.items()
        }
        # Deal with time and timestamps
        if(self.time_encoding=="cyclic"):
            for zone in zone_data:
                zone_data[zone]=self._add_time_features(zone_data[zone])
        
        self._print_df(zone_data)

        return zone_data

    def _print_df(self, zone_data):
        """Debug function to print the dataframes to an excel file."""
        output_path = os.path.join(self.root, "neiso_test_by_zone.xlsx")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer: # type: ignore
            for zone, df in zone_data.items():
                df.to_excel(writer, sheet_name=zone, index=False)

        print(f"Wrote test file to {output_path}")

    def _load_zone_year(self, file_path: str, zone: str) -> pd.DataFrame:
        df = pd.read_excel(file_path, sheet_name=zone)
        # always want Date & Hr_End
        keep = ["Date", "Hr_End"]
        # Decide which other columns to keep based on the config file:
        if self.demand_source in ("da", "both"):
            keep.append("DA_Demand")
        if self.demand_source in ("rt", "both"):
            keep.append("RT_Demand")
        if self.use_weather:
            keep += ["Dry_Bulb", "Dew_Point"]

        #Only keep the columns defined above.
        sub = df.loc[:, keep].copy()        
        return sub

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cyclic (sin, cos) encodings for hour-of-day, day-of-week and
        day-of-year and returns the same DataFrame.
        Called only when self.time_encoding == "cyclic".
        """
        # Make Hr_End numeric: 
        pd.to_numeric(df["Hr_End"], errors="raise")
        # Build timestamps from our date and hours
        ts: pd.Series = (
            pd.to_datetime(df["Date"], format="%Y-%m-%d")
            + pd.to_timedelta(df["Hr_End"] - 1, unit="h")
        )
        # Hour of day, day of week, day of year. These are are time series.
        hod = ts.dt.hour
        dow = ts.dt.dayofweek         # Monday==0 … Sunday==6
        doy = ts.dt.dayofyear         # 1 … 365/366

        # Cyclic encodings, ie we get the angle θ=2π(x/T), where x is our value. Then we pass it to sin or cos. 
        df["hod_sin"] = np.sin(2 * np.pi * hod / 24)
        df["hod_cos"] = np.cos(2 * np.pi * hod / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        #We drop the hour and date columns
        df.drop(columns=["Date", "Hr_End"], inplace=True)
        return df


    def _prepare(self) -> None:
        #Concatenate the data from the different files.
        self.data=self._load_neiso_dataframe()
        N, F, T = len(ZONES), 1, 1  # dummy: 1 time step, 1 feature
        self.A = torch.eye(N)                          # (N, N)
        self.X = torch.zeros(T, N, F)                  # (T, N, F)
        self.y = torch.zeros(T, N, self.cfg.horizon)   # (T, N, horizon)
        


    def __len__(self) -> int:
        # 1 sample (until sliding-window logic is added)
        return 1

    def __getitem__(self, idx: int):
        assert self.X is not None
        assert self.y is not None
        return self.X[0], self.A, self.y[0]



if __name__ == "__main__":

    
    # basic boilerplate config
    class NEISOConfigDict(TypedDict):
        root: str
        window: int
        horizon: int
        split: str
        demand_source: Literal["rt", "da", "both"]
        use_weather: bool
        time_encoding: Literal["cyclic", "none"]
        start_year: int
        end_year: int

    cfg_dict: NEISOConfigDict = {
        "root": "",
        "window": 12,
        "horizon": 12,
        "split": "train",
        "demand_source": "rt",
        "use_weather": True,
        "time_encoding": "cyclic",
        "start_year":2019,
        "end_year":2023
        }
    cfg = NEISOConfig(**cfg_dict)

    ds = NEISOHourlyZonal(cfg)

