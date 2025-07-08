import pandas as pd

if __name__ == "__main__":

    """ This is the code I used to turn the original data into the clean dataframe"""
    dfraw=pd.read_csv("melpeds_raw.csv")
    dfraw.columns = dfraw.columns.str.strip()  # remove any leading/trailing spaces
    
    dfraw["Datetime"] = pd.to_datetime(
    # date part
    dfraw["Sensing_Date"].astype(str)
    + " "
    # hour part, zero-filled to two digits\
    + dfraw["HourDay"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    + ":00"
)

    dfraw = dfraw.drop(columns=["Sensing_Date", "HourDay"])
    dfraw.sort_values(by="Datetime", inplace=True)
    dfraw.to_csv("melpeds_int.csv")

    df = dfraw.pivot_table(
    index="Datetime",
    columns="Sensor_Name",
    values=["Direction_1", "Direction_2"],
    aggfunc="first"
    )
    df.columns = [
    f"{sensor}_{direction}"
    for direction, sensor in df.columns
    ]
    df.columns.name = None

    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv("sensor_directions_wide.csv", index=True)