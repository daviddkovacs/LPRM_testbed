import datetime

from datacube_loader import MICROWAVE_datacube
from datacube_utilities import (mpdi, calc_Holmes_temp, frequencies, ravel_roi_time)
import pandas as pd
import matplotlib.pyplot as plt
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from lprm.retrieval.lprm_general import load_aux_file
from lprm.retrieval.lprm_v6_1.parameters import (
    get_lprm_parameters_for_frequency,
)
import xarray as xr
import numpy as np
from plot_functions import plot_hexbin, usual_stats, regressor_calc, world_map
from joblib import Parallel, delayed
import itertools
from lprm.satellite_specs import SensorSpecifics, get_specs


def load_TB_daily(bbox,time_start,time_stop,sensor ="AMSR2", file_pattern= None,
                  path = None, date_pattern= None, resolution=None):
    """
    Load day/night TBs. we need to re-assign the time dimension, as MICROWAVE_datacube assigned the average scantime
    values within bbox (skews observation times when bbox is global)
    :param bbox: List[min_lon,min_lat,max_lon,max_lat]
    :param time_start: date
    :param time_stop: date
    :param file_pattern: str AMSR2: "amsr2_l1bt_*.nc"
    :return: xr Dataset of day and night TBs with daily timestamps
    """
    nested_group_name = "S1" if sensor == "GMI" else None

    TB_DAY = MICROWAVE_datacube(bbox=bbox,
                                overpass="day",
                                time_start=time_start,
                                time_stop=time_stop,
                                sensor=sensor,
                                file_pattern=file_pattern,
                                nested_group_name=nested_group_name,
                                path_user = path,
                                date_pattern=date_pattern,
                                resolution=resolution
                                )

    TB_NIGHT = MICROWAVE_datacube(bbox=bbox,
                                  overpass="night",
                                  time_start=time_start,
                                  time_stop=time_stop,
                                  sensor=sensor,
                                  file_pattern=file_pattern,
                                  nested_group_name = nested_group_name,
                                  path_user=path,
                                  date_pattern=date_pattern,
                                  resolution=resolution
                                  )


    TB_DAY['time'] = pd.to_datetime(TB_DAY.time.dt.date.values)
    TB_NIGHT['time'] = pd.to_datetime(TB_NIGHT.time.dt.date.values)

    return TB_DAY, TB_NIGHT


def calc_MPDI_bands(TB_DAY,TB_NIGHT, list_of_bands=["l","c1","c2", "x", "ku"], minimum_mpdi = 0.01, sensor ="AMSR2"):
    """
    We calculate MPDIs for different frequencies
    :param TB_DAY: Daytime TB stack
    :param TB_NIGHT: Nighttime TB stack
    :return: Dictionary with keys as bands and values as MPDI datasets
    """

    MPDI_DAY_dict = {}
    MPDI_NIGHT_dict = {}

    for band in list_of_bands:
        try:
            _mpdi_day = mpdi(TB_DAY, band, sensor=sensor)
            MPDI_DAY_dict[band] = _mpdi_day.where(_mpdi_day>minimum_mpdi)

            _mpdi_night = mpdi(TB_NIGHT, band, sensor=sensor)
            MPDI_NIGHT_dict[band] = _mpdi_night.where(_mpdi_night>minimum_mpdi)

        except KeyError:
            print(f"{band} omitted")

    return MPDI_DAY_dict, MPDI_NIGHT_dict


def calc_MPDI_difference(MPDI_day, MPDI_night, list_of_bands=["l","c1","c2", "x", "ku"],):
    """
    We calculate the difference in MPDI. Night-Day!!!
    :param MPDI_day: MPDI calculated for daytime obs
    :param MPDI_night: MPDI calculated for nighttime obs
    :param list_of_bands: frequencies needed to calc MPDI dif for
    :return: dictionary containing list_of_bands MPDI differences
    """

    MPDI_difference_dict = {}

    for band in list_of_bands:
        try:
            MPDI_difference_dict[band] = MPDI_night[band] - MPDI_day[band]
        except KeyError:
            print(f"{band} omitted")
    return MPDI_difference_dict


def retrieve_LPRM(TB_DATASET, SURFACE_T, band, SM_input = None, VOD_input = None, sensor = "AMSR2"):
    """
    Retrieve LPRM, traditional method. Input is Brightness temps, Holmes "KA" temp and band
    :return: SM and VOD datasets
    """
    times = TB_DATASET.time
    sensor_specs = get_specs(sensor)
    inc_angle = sensor_specs.incidence_angle[0]

    band = band.upper()
    freq = sensor_specs.frequencies[band.upper()]

    lprm_list_sm = []
    lprm_list_vod = []
    lprm_list_tsim = []

    for t in times:
        try:
            print(t.dt.date.item())
            tb_map = TB_DATASET.sel(time = t).compute()
            holmes_t = SURFACE_T.sel(time = t).compute()

            if SM_input is not None:
                sm_input = SM_input.sel(time = t).compute().values
                vod_input = VOD_input.sel(time = t).compute().values
            else:
                sm_input = None
                vod_input = None
            aux_data_dict = {
                "sand": load_aux_file(0.25, "SND"),
                "clay": load_aux_file(0.25, "CLY"),
                "bld": load_aux_file(0.25, "BLD"),
            }
            params = get_lprm_parameters_for_frequency(band, inc_angle)

            sm, vod,tsim = par100.run_band(
                tb_map[f"bt_{freq}V"].values,
                tb_map[f"bt_{freq}H"].values,
                holmes_t.values,
                aux_data_dict["sand"],
                aux_data_dict["clay"],
                aux_data_dict["bld"],
                params.Q,
                params.w,
                params.opt_atm,
                inc_angle,
                params.h1,
                params.h2,
                params.vod_Av,
                params.vod_Bv,
                float(freq),
                params.temp_freeze,
                False,
                None,
                SM_map_night = sm_input,
                VOD_map_night = vod_input,
            )

            sm_da = xr.DataArray(
                data=sm,
                coords=tb_map.coords,
                dims=tb_map.dims,
                name="sm"
            )

            vod_da = xr.DataArray(
                data=vod,
                coords=tb_map.coords,
                dims=tb_map.dims,
                name="vod"
            )
            sm_da = sm_da.where(sm_da>=0)
            vod_da = vod_da.where(vod_da>=0)

            if SM_input is not None:
                tsim_da = xr.DataArray(
                    data=tsim,
                    coords=tb_map.coords,
                    dims=tb_map.dims,
                    name="tsim"
                )
                tsim_da = tsim_da.where(tsim_da>=0)
                lprm_list_tsim.append(tsim_da)

            lprm_list_sm.append(sm_da)
            lprm_list_vod.append(vod_da)
        except Exception as e:
            print(f"{e} {t.dt.date.item()}")

    SM_dataset = xr.concat(lprm_list_sm, dim = "time")
    VOD_dataset = xr.concat(lprm_list_vod, dim = "time")
    if SM_input is not None:
        TSIM_dataset = xr.concat(lprm_list_tsim, dim="time")
    else:
        TSIM_dataset = np.zeros(9)

    return SM_dataset, VOD_dataset, TSIM_dataset


def coarse_grid(DATA, resolution = 5):

    coarsen_multiplier = int(resolution / 0.25 )# If 0.25 cci grid is used for TBs

    _DATA_coarse_grid = DATA.coarsen(lat=coarsen_multiplier,
                                     lon=coarsen_multiplier,
                                     boundary="exact").construct(
        lat=("lat_grid", "lat_pixel"),
        lon=("lon_grid", "lon_pixel")
    )
    DATA_coarse_grid = _DATA_coarse_grid.assign_coords(
        lat_grid=_DATA_coarse_grid.lat.mean(dim="lat_pixel"),
        lon_grid=_DATA_coarse_grid.lon.mean(dim="lon_pixel")
    )
    return DATA_coarse_grid


def get_empty_grid(resolution, COARSE_coords):

    min_lat, max_lat = COARSE_coords.lat.min().item(), COARSE_coords.lat.max().item()
    min_lon, max_lon = COARSE_coords.lon.min().item(), COARSE_coords.lon.max().item()

    lats = np.arange(min_lat, max_lat, resolution)
    lons = np.arange(min_lon, max_lon, resolution)

    empty_data = np.full( (len(lats), len(lons)), np.nan)

    empty_grid = xr.DataArray(
        data=empty_data,
        dims=["lat", "lon"],
        coords={
            "lat": lats,
            "lon": lons
        },
        name="empty_grid"
    )
    return empty_grid


def regression_process_pixel(lat_val,
                             lon_val,
                             X_DATA,
                             Y_DATA,
                             x_var="T_KA",
                             y_var="TSIM_low_mpdi",
                             ):
    """
    Function to process regression on a single block o
    :param lat_val:
    :param lon_val:
    :param X_DATA:
    :param Y_DATA:
    :param x_var:
    :param y_var:
    :return:
    """
    X_DATA_box = X_DATA.sel(lat_grid=lat_val, lon_grid=lon_val, method="nearest")
    Y_DATA_box = Y_DATA.sel(lat_grid=lat_val, lon_grid=lon_val, method="nearest")

    df_box = pd.DataFrame({
        x_var: X_DATA_box.compute().to_numpy().ravel(),
        y_var: Y_DATA_box.compute().to_numpy().ravel(),
    }).dropna()

    result = {'lat': lat_val, 'lon': lon_val}

    if df_box.empty:
        result.update({'r': np.nan, 'rmse': np.nan, 'bias': np.nan,
                       'n': np.nan, 'slope': np.nan, 'intercept': np.nan, "ubrmse": np.nan})
        return result

    stats_box = usual_stats(df_box[x_var], df_box[y_var])

    try:
        regression_statistics = regressor_calc(df_box, x_var, y_var)
    except:
        result.update({'r': np.nan, 'rmse': np.nan, 'bias': np.nan,
                       'n': np.nan, 'slope':  np.nan, 'intercept': np.nan,"ubrmse": np.nan})
        return result

    result.update({
        'r': stats_box["r"],
        'rmse': stats_box["rmse"],
        'bias': stats_box["bias"],
        'ubrmse': stats_box["ubrmse"],
        'n': len(df_box),
        'slope': regression_statistics["m"],
        'intercept': regression_statistics["c"]
    })

    return result


def regression_wrapper(X_DATA,Y_DATA, resolution =5, bounds = [ -180,-90,180,90 ],):
    """
    This function wraps the parralel processor and its functionalities.
    :param X_DATA: X axis of the scatter (usually T_KA)
    :param Y_DATA: Y axis of the scatter (usually T_SIM_low_mpdi)
    :param resolution: in degs
    :param bounds: bbox
    :return: dataset with regression stats for pixel
    """

    X_DATA_COARSE = coarse_grid(X_DATA, resolution=resolution).compute()
    Y_DATA_COARSE = coarse_grid(Y_DATA,resolution=resolution).compute()

    empty_map = get_empty_grid(resolution=resolution, COARSE_coords = X_DATA_COARSE)

    lats = empty_map.lat[(empty_map.lat > bounds[1]) & (empty_map.lat < bounds[3])].values
    lons = empty_map.lon[(empty_map.lon > bounds[0]) & (empty_map.lon < bounds[2])].values

    coords = list(itertools.product(lats, lons))

    results_list = Parallel(n_jobs=-1)(
        delayed(regression_process_pixel)(lat.item(), lon.item(), X_DATA_COARSE, Y_DATA_COARSE)
        for lat, lon in coords
    )

    stat_da = pd.DataFrame(results_list).set_index(['lat', 'lon']).to_xarray()

    return stat_da


def get_sensor_band(TB,sensor, band, pol):

    freq = get_specs(sensor).frequencies[band.upper() if band.upper()!="KA" else "Ka"]
    _TB = TB[f"bt_{freq}{pol}"]
    return _TB


def calc_mpdi_delta(MPDI_day,
                        MPDI_night,
                        list_of_bands = ["l","c1","c2","x","ku"],
                        ):

    MPDI_ABS_DIF_dict = {}

    for band in list_of_bands:
        try:
            MPDI_ABS_DIF_dict[band] = MPDI_night[band] - MPDI_day[band]
            # MPDI_ABS_DIF_dict[band] = MPDI_day[band] - MPDI_night[band]

        except KeyError:
            print(f"{band} omitted")

    return MPDI_ABS_DIF_dict



file_pattern_lut  = {"AMSR2": "amsr2_l1bt_*.nc",
                     "SMAP": "smap_spl3smp_v009_l3bt_*.nc",
                     }
date_pattern_lut = {"AMSR2": "_(\\d{8})_",
                    "SMAP" : r"(\d{8})"}
##

if __name__=="__main__":

    bbox = [
    -124.46765153923654,
    38.280592852377055,
    -120.85139147901907,
    40.159779582899006
  ]
    year_start = "2019"
    time_start = f"{year_start}-01-01"
    time_stop = "2020-01-01"
    bandlist = ["l","c1", "c2","x", "ku"]
    sensor = "AMSR2"
    resolution = "medium_resolution"

    TB_DAY, TB_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                     sensor=sensor,file_pattern=file_pattern_lut[sensor],
                                     date_pattern=date_pattern_lut[sensor],
                                     resolution=resolution
                                     )

    MPDI_DAY , MPDI_NIGHT = calc_MPDI_bands(TB_DAY=TB_DAY,TB_NIGHT=TB_NIGHT,
                                            list_of_bands=bandlist,
                                            minimum_mpdi=0.01,
                                            sensor=sensor)

    MPDI_delta = calc_mpdi_delta(MPDI_day=MPDI_DAY,
                                         MPDI_night=MPDI_NIGHT,
                                         )
##
    for ms in range(1,13):
        plot_t_start = datetime.date(2019, ms, 1)
        plot_t_end = datetime.date(2019, ms, 10)
        current_band = "x"

        MPDI_avg = MPDI_delta[current_band].sel(time=slice(plot_t_start,plot_t_end)).sum(dim="time").compute()
        MPDI_abs = abs(MPDI_avg)
        MPDI_abs.plot(
            # vmin=0,
            # vmax = 0.2,
            cmap="viridis")
        plt.title(f"{current_band.upper()}-band MPDI_night - MPDI_day ({plot_t_start.year} {plot_t_start.month})")
        plt.show()