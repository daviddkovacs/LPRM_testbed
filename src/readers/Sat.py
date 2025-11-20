import os.path
import matplotlib
matplotlib.use("TkAgg")
from datetime import datetime
import xarray as xr
import pandas as pd
import os
from lprm.satellite_specs import get_specs

def resolution_path(target_res):

    if target_res == "10":
        res_path = "medium_resolution"
    elif target_res == "25":
        res_path = "coarse_resolution"

    return res_path


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

        self.path = path
        self.sat_freq = get_specs(sat_sensor.upper()).frequencies[sat_freq.upper()]
        self.sat_sensor = sat_sensor
        self.overpass = overpass
        self.target_res = target_res

        res_path = resolution_path(target_res)

        year_month = date.strftime("%Y%m")
        date_fmt = date.strftime("%Y%m%d")
        pattern = f"{sat_sensor}_l1bt_{overpass}_{date_fmt}_{target_res}km.nc"

        self.bt_file = os.path.join(path, overpass,year_month,pattern)


    def to_pandas(self):

        dataset = self.to_xarray()
        pandas = dataset.to_dataframe()
        # pandas = pandas.dropna(subset=['scantime']).reset_index()
        # pandas = pandas.reset_index()
        pandas = pandas[["SCANTIME", f"BT_{self.sat_freq}V", f"BT_{self.sat_freq}H"]]
        pandas = pandas.rename(columns={f"BT_{self.sat_freq}V": "BT_V",
                                        f"BT_{self.sat_freq}H": "BT_H",})
        pandas.columns = pandas.columns.str.upper()
        pandas = pandas.droplevel("TIME")

        return pandas


    def to_xarray(self,
                  bbox= None):

        dataset = xr.open_dataset(self.bt_file, decode_timedelta=False)
        dataset = dataset.rename({v: v.upper() for v in dataset.variables})
        dataset = dataset.rename({v: v.upper() for v in dataset.dims})

        if "time" in dataset.dims:
            dataset = dataset.squeeze("time", drop=True)
        if bbox:
            lat_mask = (dataset["LAT"] >= bbox[1]) & (dataset["LAT"] <= bbox[3])
            lon_mask = (dataset["LON"] >= bbox[0]) & (dataset["LON"] <= bbox[2])
            dataset = dataset.where(lat_mask & lon_mask, drop=True)

        if dataset['LAT'].ndim == 2:

            lat_1d = dataset['LAT'][:, 0]
            lon_1d = dataset['LON'][:, 0]

            ds = dataset.drop_vars(['LAT', 'LON'])
            dataset = ds.assign_coords(LAT=lat_1d, LON=lon_1d)

        else:
            dataset = dataset.assign_coords(
                {"LAT" : dataset["LAT"],
                 "LON" : dataset["LON"]})

        return dataset


class LPRMData(BTData):

    def __init__(self, path, sat_sensor, date, overpass, target_res, sat_freq=None):

        year_month = date.strftime("%Y%m")
        date_fmt = date.strftime("%Y%m%d")
        pattern = f"{sat_sensor.upper()}_LPRM_VEGC_{overpass}{date_fmt}_{target_res}km_v061.nc"
        res_path = resolution_path(target_res)

        bt_file = os.path.join(path, overpass,year_month,pattern)

        super().__init__(path, sat_sensor, date, overpass, target_res, sat_freq)
        self.bt_file = bt_file


    def to_pandas(self):

        dataset = self.to_xarray()
        pandas = dataset.to_dataframe()

        return pandas