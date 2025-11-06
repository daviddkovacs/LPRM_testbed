import os.path
import rioxarray
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import pandas as pd
import os

def clms_filepattern(path, date, bio_var):

    bio_var = bio_var.upper()
    if (date == "2024-10-22" or date == '2024-10-25'):
        _date = "20241020"
    elif date == '2024-10-31':
        _date = '20241031'
    else:
        print(f"Warning! CLMS date not found for input date: {date}")

    pattern = f"CLMS_{bio_var.upper()}_{_date}.nc"
    bio_file = os.path.join(path, bio_var.upper(), pattern)

    return bio_file


def era5_filepattern(path, date,  time = "1200"):
    """
    get ERA5 specific filenames and directory.
    :return: full path for full ERA5 daily file..
    """

    year = datetime.strptime(date, "%Y-%m-%d").strftime("%Y")
    ddd = datetime.strptime(date, "%Y-%m-%d").strftime("%j")
    fulldate = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

    pattern = f"ERA5-LAND_AN_{fulldate}_{time}.nc"
    bio_file = os.path.join(path, year, ddd, pattern)

    return bio_file

def era5_coords_converter(longitude):
    """
    Converting ERA5 longitudes from its grid to range b/w -180 and 180 deg
    """
    array = (longitude + 180) % 360 - 180

    return array


class Bio:
    """
    Bio-geophysical data reader, either from ERA5 or CLMS data.
    """
    def __init__(self,
                 path,
                 date,
                 bio_var,
                 time="1200"):

        self.path = path
        self.bio_var = bio_var

        self.bio_file = era5_filepattern(path,date)

    def to_pandas(self, bbox=None):
        """
        Converting to dataframe. Optionally, can be passed a bounding box, to restrict reading too much data into memory.
        :return: pd.DataFrame
        """
        if bbox is None:
            bbox = list([-120.66213,32.3257314,-88.03080,41.79186])

        dataset = self.to_xarray(bbox)
        pandas = dataset.to_dataframe()
        pandas = pandas.dropna(subset=[self.bio_var]).reset_index()

        pandas = pandas[["lon", "lat", f"{self.bio_var}"]]

        return pandas

    def to_xarray(self,bbox):
        """
        Creating dask and then filtered xarray. Works for both ERA5 and CLMS
        :return: xr.DataSet
        """
        dataset = xr.open_dataset(self.bio_file, chunks="auto",
                                  decode_timedelta=False)

        if "longitude" in dataset.coords:
            dataset = dataset.rename({"longitude": "lon",
                                            "latitude": "lat", })
            dataset["lon"] = era5_coords_converter(dataset["lon"])

        lons = dataset["lon"]
        lats = dataset["lat"]

        dataset = dataset.loc[
            dict(lon = (lons > bbox[0]) & (lons < bbox[2]),
                 lat = (lats > bbox[1]) & (lats < bbox[3]))
        ]

        # dataset = dataset.squeeze("time", drop=True)
        return dataset

class CLMS(Bio):
    """
    CLMS Data reader. Child of Bio, needed to read LAI and FCOVER data.
    """
    def __init__(self, path, date, bio_var,):

        self.bio_file = clms_filepattern(path,date,bio_var)

        super().__init__( path, date, bio_var,)

