import os.path
import rioxarray
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import pandas as pd
import os

class ERA:
    def __init__(self,
                 path,
                 date,
                 bio_var,
                 time="1200"):

        self.path = path
        self.date = date
        self.bio_var = bio_var

        year = datetime.strptime(date, "%Y-%m-%d").strftime("%Y")
        ddd = datetime.strptime(date, "%Y-%m-%d").strftime("%j")
        fulldate = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

        pattern = f"ERA5-LAND_AN_{fulldate}_{time}.nc"
        self.era_file = os.path.join(path,year,ddd,pattern)


    def to_pandas(self):

        dataset = self.to_xarray()
        pandas = dataset.to_dataframe()
        pandas = pandas.dropna(subset=[self.bio_var]).reset_index()
        pandas["longitude"] = (pandas["longitude"] + 180) % 360 - 180
        pandas = pandas[["longitude", "latitude", f"{self.bio_var}"]]
        pandas = pandas.rename(columns={"longitude": "lon",
                                        "latitude": "lat", })
        return pandas

    def to_xarray(self,):

        dataset = xr.open_dataset(self.era_file, decode_timedelta=False)
        # dataset = dataset.squeeze("time", drop=True)

        return dataset
