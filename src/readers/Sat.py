import os.path
import matplotlib
matplotlib.use("TkAgg")
from datetime import datetime
import xarray as xr
import pandas as pd
import os

class BTData:
    """
    Class to read in Satellite data from (currently from AMSR2)

    path: location of .nc files
    sat_sensor: satellite sat_sensor used (currently amsr2)
    date: in format YYYY-MM-DD
    overpass: day, night
    target_res: pixel resolution 10 or 25 (km)
    frequency: 6.9, 7.3, 10.7, 18.7, 23.8, 36.5, 89.0
    """
    def __init__(self,
                 path,
                 sat_sensor,
                 date,
                 overpass,
                 target_res,
                 sat_freq,
                 *args,
                 **kwargs):

        self.sat_freq = sat_freq
        self.sat_sensor = sat_sensor
        self.overpass = overpass
        self.target_res = target_res

        year_month = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m")
        date_fmt = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

        pattern = f"{sat_sensor}_l1bt_{overpass}_{date_fmt}_{target_res}km.nc"

        self.bt_file = os.path.join(path,overpass,year_month,pattern)


    def to_pandas(self):

        dataset = self.to_xarray()
        pandas = dataset.to_dataframe()
        pandas = pandas.dropna(subset=['scantime']).reset_index()
        pandas = pandas[["lon","lat","scantime", f"bt_{self.sat_freq}V", f"bt_{self.sat_freq}H"]]
        pandas = pandas.rename(columns={f"bt_{self.sat_freq}V": "bt_V",
                                        f"bt_{self.sat_freq}H": "bt_H",})
        return pandas


    def to_xarray(self,
                  ):

        dataset = xr.open_dataset(self.bt_file, decode_timedelta=False)
        if "time" in dataset.dims:
            dataset = dataset.squeeze("time", drop=True)

        return dataset


class LPRMData(BTData):

    def __init__(self, path, sat_sensor, date, overpass, target_res, sat_freq=None):

        year_month = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m")
        date_fmt = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
        pattern = f"{sat_sensor.upper()}_LPRM_VEGC_{overpass}{date_fmt}_{target_res}km_v061.nc"
        bt_file = os.path.join(path, overpass, year_month, pattern)

        super().__init__(path, sat_sensor, date, overpass, target_res, sat_freq)
        self.bt_file = bt_file


    def to_pandas(self):

        dataset = self.to_xarray()
        pandas = dataset.to_dataframe()
        pandas.columns = pandas.columns.str.lower()
        pandas = pandas.dropna(subset=['scantime'])

        return pandas