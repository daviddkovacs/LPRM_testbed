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


def load_TB_daily(bbox,time_start,time_stop,sensor ="AMSR2"):
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
    file_pattern = f"{sensor.lower()}_l1bt_*.nc"

    TB_DAY = MICROWAVE_datacube(bbox=bbox,
                                overpass="day",
                                time_start=time_start,
                                time_stop=time_stop,
                                sensor=sensor,
                                file_pattern=file_pattern,
                                nested_group_name=nested_group_name,
                                )

    TB_NIGHT = MICROWAVE_datacube(bbox=bbox,
                                  overpass="night",
                                  time_start=time_start,
                                  time_stop=time_stop,
                                  sensor=sensor,
                                  file_pattern=file_pattern,
                                  nested_group_name = nested_group_name,
    )


    TB_DAY['time'] = pd.to_datetime(TB_DAY.time.dt.date.values)
    TB_NIGHT['time'] = pd.to_datetime(TB_NIGHT.time.dt.date.values)

    return TB_DAY, TB_NIGHT


def calc_MPDI_bands(TB_DAY,TB_NIGHT, list_of_bands=["c1","c2", "x", "ku"], minimum_mpdi = 0.01):
    """
    We calculate MPDIs for different frequencies
    :param TB_DAY: Daytime TB stack
    :param TB_NIGHT: Nighttime TB stack
    :return: Dictionary with keys as bands and values as MPDI datasets
    """

    MPDI_DAY_dict = {}
    MPDI_NIGHT_dict = {}

    for band in list_of_bands:
        _mpdi_day = mpdi(TB_DAY,band)
        MPDI_DAY_dict[band] = _mpdi_day.where(_mpdi_day>minimum_mpdi)

        _mpdi_night =  mpdi(TB_NIGHT,band)
        MPDI_NIGHT_dict[band] = _mpdi_night.where(_mpdi_night>minimum_mpdi)

    return MPDI_DAY_dict, MPDI_NIGHT_dict


def calc_MPDI_difference(MPDI_day, MPDI_night, list_of_bands=["c2", "x", "ku"]):
    """
    We calculate the difference in MPDI. Night-Day!!!
    :param MPDI_day: MPDI calculated for daytime obs
    :param MPDI_night: MPDI calculated for nighttime obs
    :param list_of_bands: frequencies needed to calc MPDI dif for
    :return: dictionary containing list_of_bands MPDI differences
    """

    MPDI_difference_dict = {}

    for band in list_of_bands:
        MPDI_difference_dict[band] = MPDI_night[band] - MPDI_day[band]
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


def get_empty_grid(resolution):

    lats = np.arange(-87.5, 90, resolution)
    lons = np.arange(-177.5, 180, resolution)

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
                             global_slope = 0.85,
                             global_intercept = 58.07,
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

    empty_map = get_empty_grid(resolution=resolution)

    lats = empty_map.lat[(empty_map.lat > bounds[1]) & (empty_map.lat < bounds[3])].values
    lons = empty_map.lon[(empty_map.lon > bounds[0]) & (empty_map.lon < bounds[2])].values

    coords = list(itertools.product(lats, lons))

    results_list = Parallel(n_jobs=-1)(
        delayed(regression_process_pixel)(lat.item(), lon.item(), X_DATA_COARSE, Y_DATA_COARSE)
        for lat, lon in coords
    )

    stat_da = pd.DataFrame(results_list).set_index(['lat', 'lon']).to_xarray()

    return stat_da


##
if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    year_start = "2018"
    time_start = f"{year_start}-01-01"
    time_stop = "2019-01-01"
    bandlist = ["c1","c2", "x", "ku"]
    sensor = "GMI"


    TB_DAY, TB_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                     sensor=sensor,
                                     )
    HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(TB_NIGHT, sensor=sensor), calc_Holmes_temp(TB_DAY, sensor=sensor)

##
    band_current = "ku"
    SM_NIGHT, VOD_NIGHT,_ = retrieve_LPRM(TB_DATASET=TB_NIGHT, SURFACE_T=HOLMES_T_NIGHT, band=band_current, sensor=sensor)
    # Highly experimental! TSIM is obtained byrunning LPRM in reverse.
    # TB has to be corresponding, for T_SIM to work!!!! DAY-DAY NIGHT-NIGHT
    # SURFACE_T doesnt matter if SM_input is True.
    _, _, TSIM_DAY = retrieve_LPRM(TB_DATASET=TB_DAY, SURFACE_T=HOLMES_T_DAY, band=band_current,
                                   SM_input=SM_NIGHT, VOD_input=VOD_NIGHT, sensor=sensor)

##
    dif_threshold = 0.00005
    minimum_mpdi = 0.01

    MPDI_DAY , MPDI_NIGHT = calc_MPDI_bands(TB_DAY=TB_DAY,TB_NIGHT=TB_NIGHT,
                                            list_of_bands=bandlist, minimum_mpdi=minimum_mpdi)
    MPDI_deltas = calc_MPDI_difference(MPDI_day=MPDI_DAY,
                                       MPDI_night=MPDI_NIGHT,
                                       list_of_bands=bandlist)

    MPDI_DELTA_band = MPDI_deltas[band_current]

    low_mpdi_mask = xr.where((MPDI_DELTA_band >= -dif_threshold) & (MPDI_DELTA_band <= dif_threshold),
                             1, 0).compute()

    SM_low_mpdi = xr.where((low_mpdi_mask==1),SM_NIGHT,np.nan)
    VOD_low_mpdi = xr.where((low_mpdi_mask==1),VOD_NIGHT,np.nan)
    TB_DAY_low_mpdi = xr.where((low_mpdi_mask==1),TB_DAY,np.nan)
    TSIM_low_mpdi = xr.where((low_mpdi_mask==1),TSIM_DAY,np.nan)

    T_KA = TB_DAY_low_mpdi["bt_36.5V"]

##
    res = 1
    stat_da = regression_wrapper(T_KA,TSIM_low_mpdi,resolution=res)

##
    world_map(stat_da, "intercept", cbar_min=0,cbar_max=100, cmap="viridis", title_extra = f"{year_start} {band_current}")
    world_map(stat_da, "slope", cbar_min=0.8,cbar_max=1.1, cmap="RdYlGn",title_extra = f"{year_start} {band_current}")
    world_map(stat_da, "r", cbar_min=0.5,cbar_max=1, cmap="coolwarm",title_extra = f"{year_start} {band_current}")
    world_map(stat_da, "rmse", cbar_min=0,cbar_max=25, cmap="YlGn",title_extra = f"{year_start} {band_current}")
    world_map(stat_da, "bias", cbar_min=0,cbar_max=25, cmap="Purples",title_extra = f"{year_start} {band_current}")
    world_map(stat_da, "ubrmse", cbar_min=0,cbar_max=5, cmap="Purples",title_extra = f"{year_start} {band_current}")

##
    highres_coords = HOLMES_T_DAY.isel(time =0)
    _stat_da = stat_da.reindex_like(highres_coords, method="nearest" )
    compression_settings = {"zlib": True, "complevel": 5}

    encoding_dict = {"sm": compression_settings}

    # _stat_da.to_netcdf("/home/ddkovacs/personal_data/lprm_daytime/lprm_testing/T_aux/"
    #                  f"Daytime_T_aux_{band_current}.nc", encoding={key : compression_settings for key in stat_da.var()})

##
    _density_plot_rois = {
        "sahel":
            [
                -14.164029114749468,
                7.731771192012118,
                9.907581803277111,
                13.006449666672083
            ],
        "global":
            [-180, -90, 180, 90]
        ,
        "sahara":
            [
                -11.088079487820153,
                14.199665315164381,
                12.8292385543796,
                22.63947012882589
            ]
        ,
        "amazon":
            [
                -71.7768365566669,
                -8.491843421295215,
                -67.95189966183453,
                -5.472202615630479
            ]
        ,
        "mississippi":
            [
                -94.9768726933208,
                30.214798554771548,
                -91.7883126223943,
                34.546414390488565
            ]
        ,
        "deciduous_w_virginia":
            [
                -84.49150294626182,
                34.757704980926704,
                -78.90004665477167,
                40.778886369880695
            ]
        ,
        "australia":
            [
                144.04056490447869,
                -35.50692205951431,
                147.3312456957429,
                -33.30894507250183
            ]

    }

    region = "sahara"
    roi = _density_plot_rois[region]

    T_KA = TB_DAY_low_mpdi["bt_36.5V"]
    T_HOLMES = T_KA * 0.893 + 44.8
    DELTA_T = TSIM_low_mpdi - T_KA

    F = (TB_DAY_low_mpdi[f"bt_{frequencies["ku".upper()]}H"]
         /TB_DAY_low_mpdi[f"bt_{frequencies["ka".upper()]}V"])

    date_range = pd.date_range(start=time_start,end=time_stop,freq="MS")

    for i in date_range.year:
        # time_selector = (DELTA_T.time.dt.month == i)
        time_selector = (DELTA_T.time.dt.year == int(year_start))
        df = pd.DataFrame({
            "DELTA_T": ravel_roi_time(DELTA_T, roi, time_selector, method="nearest"),
            "F": ravel_roi_time(F, roi, time_selector, method="nearest"),
            "T_KA": ravel_roi_time(T_KA, roi, time_selector, method="nearest"),
            "TSIM_low_mpdi" : ravel_roi_time(TSIM_low_mpdi, roi, time_selector, method="nearest"),
            "VOD_low_mpdi": ravel_roi_time(VOD_low_mpdi, roi, time_selector, method="nearest"),
            "SM_low_mpdi": ravel_roi_time(SM_low_mpdi, roi, time_selector, method="nearest"),
            "T_HOLMES": ravel_roi_time(T_HOLMES, roi, time_selector, method="nearest"),
        })

        plot_hexbin(df,
                    "T_KA",
                    "TSIM_low_mpdi",
                    color_of_points="F",
                    # xlim=[0.95,1.05], ylim=[265,320],   #F
                    xlim=[265,320], ylim=[265,320],          # T and T
                    # xlim=[None,None], ylim=[None,None],
                    # cbar_min= 0, cbar_max= 30,
                    title_string=f"year:{i} {region} band: {band_current}",
                    )

