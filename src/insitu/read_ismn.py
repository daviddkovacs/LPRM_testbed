import datetime
import pandas as pd
from ismn.interface import ISMN_Interface
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



class ISMNProcessor:
    def __init__(self, ismn_data, network_of_interest):
        self.ismn_data = ismn_data
        self.network_of_interest = network_of_interest

    def filter_stations(self, bbox, lc_class):
        conditions = (
            (self.ismn_data.metadata['variable'].val == 'soil_temperature') &
            (self.ismn_data.metadata['instrument'].depth_to <= 1) &
            (self.ismn_data.metadata['network'].val <= self.network_of_interest) &
            (self.ismn_data.metadata['lc_2010'].val == lc_class) &
            (self.ismn_data.metadata['longitude'].val > bbox[0]) &
            (self.ismn_data.metadata['latitude'].val > bbox[1]) &
            (self.ismn_data.metadata['longitude'].val < bbox[2]) &
            (self.ismn_data.metadata['latitude'].val < bbox[3])
        )
        return self.ismn_data.metadata[conditions].index.to_list()

    def get_station_names(self, stations):
        _, meta = self.ismn_data.read(stations, return_meta=True)
        return np.unique(meta.xs("station"))

    def get_xr(self, station_name, nw):
        return self.ismn_data[nw][station_name].to_xarray()

    def plot_ts(self, da, station, start, end):
        ts_ = da["soil_temperature"]
        ts = ts_.dropna(dim="sensor", how="all")
        ts_clean = ts.where(ts > -100)

        lat = da.attrs["lat"]
        lon = da.attrs["lon"]

        ts_days = ts_clean.sel(date_time=slice(start, end))

        ts_days.plot.line(hue='sensor', alpha=0.7, figsize=(10, 5))
        plt.legend(da["depth_to"].values, title="Eff. depth (m)")
        plt.title(f"Soil Temperature {station} lat:{lat} lon:{lon}")
        plt.show()


    def plot_map_stations(self):

        plt.plot(figsize=(30, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ismn_data.plot_station_locations('soil_temperature', min_depth=0.5,
                                              extent=[-125.463376,-71.801765,24.241511,49.139843],markersize=5, text_scalefactor=3)
        plt.show()


    def main_runner(self, start_date, end_date, bbox, lc_class, network):
        station_ids = self.filter_stations(bbox, lc_class)
        station_list = self.get_station_names(station_ids)

        for station in station_list:
            dat = self.get_xr(station, network)
            self.plot_ts(dat, station, start_date, end_date)


if __name__ == "__main__":

    data_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/ismn_data/ ISMN_data_bulk"
    ISMN = ISMN_Interface(data_path, parallel=False)
    network_of_interest = "SCAN"

    desert_bbox = [-118.118617, 35.791179, -113.251638, 42.128318]
    desert_class = 130

    forest_bbox = [-90.230704, 30.682585, -85.621388, 34.303626]
    forest_class = 70

    start = datetime.date(2024, 4, 1)
    end = datetime.date(2024, 6, 1)

    processor = ISMNProcessor(ismn_data=ISMN, network_of_interest=network_of_interest)

    processor.main_runner(
        start_date=start,
        end_date=end,
        bbox=forest_bbox,
        lc_class=forest_class,
        network=network_of_interest
    )
    processor.plot_map_stations()

