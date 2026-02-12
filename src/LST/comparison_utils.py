import glob
import re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from typing import Literal, List
import pandas as pd
from config.paths import SLSTR_path, path_bt


# ---------------------------------------
# DATACUBE PROCESSORS
def temporal_subset_dc(SLSTR, AMSR2, date):
    """
    Select the closest date to SLSTR, and thus select this date to access AMSR2
    """
    # SLSTR["time"] = SLSTR.time.sortby("time")
    SLSTR = SLSTR.drop_duplicates(dim="time")
    SLSTR_obs = SLSTR.sel(time=date, method="nearest")

    # We select SLSTR's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=SLSTR_obs.time.dt.floor("d"), method="nearest")

    return {"SLSTR": SLSTR_obs, "AMSR2": AMSR2_obs}


def spatial_subset_dc(SLSTR, AMSR2,  bbox):
    """
    SLSTR is cut to the full spatial extent of AMSR2.
    Both AMSR2 and SLSTR cropped to bbox
    """
    AMSR2 = crop2roi(AMSR2, bbox)
    res = AMSR2.attrs["resolution"]

    AMSR2_bbox = [get_edges(AMSR2.lon.values,res).min(),
                  get_edges(AMSR2.lat.values,res).min(),
                  get_edges(AMSR2.lon.values,res).max(),
                  get_edges(AMSR2.lat.values,res).max()]

    SLSTR_roi = crop2roi(SLSTR, AMSR2_bbox)

    return {"SLSTR": SLSTR_roi, "AMSR2": AMSR2}


def SLSTR_AMSR2_datacubes(region : Literal["sahel", "siberia", "midwest","ceu"],
                          bbox= List[float],
                           SLSTR_path = SLSTR_path,
                           AMSR2_path = path_bt,
                           time_start = "2024-01-01",
                           time_stop = "2025-01-01",
                           ):
    """
    Main function to obtain SLSTR and AMSR2 observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param SLSTR_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param AMSR2_path: Path where AMSR2 brightness temperatures are stored
    :param region: Region of SLSTR. Currently downloaded: Sahel, Siberia and US Midwest
    :return: dictionary with SLSTR and AMSR2 datacubes.
    """
    SLSTR_path_region = os.path.join(SLSTR_path,region)

    SLSTR_cropped_stack = open_sltsr(SLSTR_path_region,
                             subdir_pattern = f"S3?_SL_2_LST____*",
                             date_pattern = r'___(\d{8})T(\d{4})',
                             time_start = time_start,
                             time_stop = time_stop,
                             bbox=bbox
                             )

    AMSR2_cropped_stack = open_amsr2(path=AMSR2_path,
                       sensor="AMSR2",
                       overpass="day",
                       subdir_pattern=f"20*",
                       file_pattern="amsr2_l1bt_*.nc",
                       date_pattern=r"_(\d{8})_",
                       time_start=time_start,
                       time_stop=time_stop,
                       resolution = "coarse_resolution",
                       bbox=bbox
                       )


    return  {"SLSTR" : SLSTR_cropped_stack.compute(), "AMSR2" : AMSR2_cropped_stack.compute(),}





# ---------------------------------------
# RAW SATELLITE PROCESSORS
def open_amsr2(path,
               sensor,
               overpass,
               date_pattern,
               subdir_pattern,
               file_pattern,
               resolution: Literal["coarse_resolution","medium_resolution"],
               time_start = "2024-01-01",
               time_stop = "2025-01-01",
               bbox = List[float]
               ):

    folder = os.path.join(path,resolution,sensor,overpass,subdir_pattern,file_pattern)

    files = glob.glob(folder)

    dates_string =  [re.search(date_pattern, p).group(1) for p in files]

    _dates = pd.to_datetime(dates_string)

    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))
    files_valid = np.array(files)[date_mask]

    dataset = xr.open_mfdataset(files_valid,
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto",
                                decode_timedelta = False).assign_coords(time = _dates[date_mask])

    res_dict = {"coarse_resolution" : 0.25,
                "medium_resolution":  0.1}
    dataset = dataset.assign_attrs(resolution = res_dict[resolution])
    print(f"Loading dataset finished (AMSR2)")

    return crop2roi(dataset,bbox)


def open_date(lst,
              anc,
              cloud, # "bayes_in"
              coord,
              day):

    LST = xr.open_mfdataset(lst,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)[["LST"]]

    ANC = xr.open_mfdataset(anc,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)[["NDVI","biome"]]
    NDVI = ANC["NDVI"]

    SNOWICE = ANC["biome"]
    CLOUD = xr.open_mfdataset(cloud,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)["bayes_in"]

    COORDS = xr.open_mfdataset(coord,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)[["latitude_in","longitude_in"]]

    DATA = xr.merge([NDVI,LST])[["LST","NDVI"]]

    DATA = DATA.assign_coords(
        lat=(( "rows", "columns"), COORDS.latitude_in.values),
        lon=(( "rows", "columns"), COORDS.longitude_in.values)
    )


    cloudy = xr.where(CLOUD == 2, True, False) # TODO
    CLOUD_FILTERED = xr.where(cloudy, np.nan, DATA)

    snowy = xr.where(SNOWICE==27, True, False)
    CLOUD_SNOW_FILTERED = xr.where(snowy, np.nan, CLOUD_FILTERED)

    print(f"Succesfully read SLSTR: {day}")

    return CLOUD_SNOW_FILTERED


def clean_pad_data(list_of_da):
    """
    filters empy cropped data from SLSTR as well as pads data so each cropped ROI is the same dm
    :param list_of_da: list of dataarrays to be concatenated.
    :return: list of nicely cropped dataarrays
    """
    cleaned_data = [
        ds for ds in list_of_da
        if ds.sizes['rows'] > 0 and ds.sizes['columns'] > 0
    ]
    max_rows = max(ds.sizes['rows'] for ds in cleaned_data)
    max_cols = max(ds.sizes['columns'] for ds in cleaned_data)

    padded_data = []
    for ds in cleaned_data:
        pad_rows = max_rows - ds.sizes['rows']
        pad_cols = max_cols - ds.sizes['columns']

        ds_padded = ds.pad(
            rows=(0, pad_rows),
            columns=(0, pad_cols),
            constant_values=np.nan
        )
        padded_data.append(ds_padded)

    return padded_data


def open_sltsr(path,
               subdir_pattern,
               date_pattern,
               bbox,
               time_start="2024-01-01",
               time_stop="2025-01-01",
               ):
    """
    Sorry to all my colleauges and to people who like elegant code, I also do.
    However, reading SLSTR files, with multi-year timespan in an elegant way, did not work.
    """
    folder_lst = os.path.join(path,subdir_pattern,"LST_in.nc")
    folder_anc = os.path.join(path,subdir_pattern,"LST_ancillary_ds.nc")
    folder_cloud = os.path.join(path,subdir_pattern,"flags_in.nc")
    coord_path = os.path.join(path,subdir_pattern,"geodetic_in.nc")

    files_lst = glob.glob(folder_lst)
    files_anc = glob.glob(folder_anc)
    files_cloud = glob.glob(folder_cloud)
    geo_files = glob.glob(coord_path)

    dates_string =  [(re.search(date_pattern, p).group(1),
                      re.search(date_pattern, p).group(2))for p in files_lst] # could be also files_ndvi (date comes from fname)

    _dates = pd.to_datetime([f"{dt[0]} {dt[1]}" for dt in dates_string])

    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    files_valid_lst = np.array(files_lst)[date_mask]  # LST
    files_valid_anc = np.array(files_anc)[date_mask]  # Ancillary (NDVI, Snow and Ice flags)
    files_valid_cloud = np.array(files_cloud)[date_mask]  # Cloud classification
    geo_files_valid = np.array(geo_files)[date_mask] # COORDS
    dates_valid = np.array(_dates)[date_mask] # days

    big_data = []

    for lst, anc, cloud, coord, day in zip(files_valid_lst,
                                           files_valid_anc,
                                           files_valid_cloud,
                                           geo_files_valid,
                                           dates_valid):

        daily_da = open_date(lst, anc, cloud, coord, day)
        cropped_daily_da=  crop2roi(daily_da, bbox=bbox)
        big_data.append(cropped_daily_da)

    padded_data = clean_pad_data(big_data)

    _dataset = xr.concat(padded_data, dim = "time")
    _dataset = filter_empty_var(_dataset)

    return _dataset.sortby("time")




# ---------------------------------------
# MISCELLANEOUS UTILS
frequencies = {'C1': 6.9, 'C2': 7.3, 'X': 10.7, 'KU': 18.7, 'K': 23.8, 'KA': 36.5}

def calc_Holmes_temp(KaV):
    """
    Surface temperature from Ka-band observations according to Holmes et al. 2008
    """
    TSURF = KaV["bt_36.5V"] * 0.893 + 44.8
    TSURF.attrs = KaV.attrs
    return TSURF


def calc_adjusted_temp(AMSR2, factor = 0.6, bandH = "Ka", mpdi_band = "C1"):
    """
    Theoretical MPDI adjusted temperature. Allows for free frequency selection.
    """
    _mpdi = xr.where(
        (mpdi(AMSR2, mpdi_band)<=0.05) & (mpdi(AMSR2, mpdi_band)>=0), # Apply only where MPDI is lte 0.05
        mpdi(AMSR2, mpdi_band),
        0.05)
    # _factor = xr.where(
    #     mpdi(AMSR2, mpdi_band)<=0.01,
    #     0.8,
    #     0.05)
    Teff = ((0.893 * AMSR2[f"bt_{frequencies[bandH.upper()]}H"]) /
            (1 - _mpdi / factor)) + 44.8
    return Teff


def KuKa(AMSR2, num = "Ku",denom = "Ka"):
    """
    Calculate ratio, as seen in SSM/I Cal/Val document
    https://apps.dtic.mil/sti/tr/pdf/ADA274626.pdf
    https://www.tandfonline.com/doi/epdf/10.1080/014311698215603?needAccess=true
    """
    return (AMSR2[f"bt_{frequencies[num.upper()]}H"] / AMSR2[f"bt_{frequencies[denom.upper()]}V"])


def mpdi(AMSR2, band):
    """
    calculate MPDI for AMSR2 BTs. Also accepts frequencies, to select band.
    """
    btv, bth = AMSR2[f"bt_{frequencies[band.upper()]}V"], AMSR2[f"bt_{frequencies[band.upper()]}H"]
    return ((btv-bth)/(btv+bth))


def threshold_ndvi(lst, ndvi, soil_range=[0,0.3], ndvi_range=[0.3,1]):
    """
    Simple thresholding of Soil-Veg to get different temps.
    """
    veg_temp = xr.where((max(ndvi_range)>= ndvi) & (ndvi >min(ndvi_range)), lst, np.nan)
    soil_temp = xr.where((max(soil_range)> ndvi) & (ndvi >=min(soil_range)), lst, np.nan)

    return soil_temp, veg_temp


def crop2roi(ds,bbox):
    """
    Cropping to bbox. Handles S3 projection (lat, lon) for each coord
    :param ds:
    :param bbox:
    :return:
    """
    mask = (
            (ds.lon >= bbox[0]) & (ds.lon <= bbox[2]) &
            (ds.lat >= bbox[1]) & (ds.lat <= bbox[3])
    )
    return ds.where(mask, drop=True)


def filternan(array):
    return  array.values.flatten()[~np.isnan(array.values.flatten())]


def clip_swath(ds):
    """
    Some S3 Tiles have a larger (1-2pix) across-scan dim resulting in errors. Thus we crop it.
    """
    return ds.isel(rows=slice(0, 1200))


def filter_empty_var(ds, var = "NDVI"):
    """
    Sometimes NDVI is empty.. then we filter the whole dataset
    """
    valid = ds[var].notnull().any(dim = ["rows","columns"])
    return ds.sel(time=valid)


def subset_statistics(array):

    _array = filternan(array)
    stat_dict = {}
    if np.any(~np.isnan(_array)):
        stat_dict["mean"] = np.nanmean(_array).item()
        stat_dict["std"] = np.nanstd(_array).item()
    else:
        stat_dict["mean"] = np.nan
        stat_dict["std"] = np.nan
    return _array, stat_dict





def get_edges(centers, res):
    """
    Calculate the spacing between pixels, to properly handle np.digitize. Otherwise offset.
    """

    edges = np.append(np.sort(centers) - res / 2, np.sort(centers)[-1] + res / 2)

    return np.sort(edges)


def binning_smaller_pixels(slstr_da, amsr2_da):
    res = amsr2_da.attrs["resolution"]
    lat_edges = get_edges(amsr2_da.lat.values, res)
    lon_edges = get_edges(amsr2_da.lon.values, res)

    iterables = {}

    iterables["lats"] = np.digitize(slstr_da.lat.values, lat_edges) - 1
    iterables["lons"] = np.digitize(slstr_da.lon.values, lon_edges) - 1

    return iterables


def slstr_pixels_in_amsr2(slstr_da,
                          bin_dict,
                          target_lat_bin,
                          target_lon_bin):

    mask = (bin_dict["lats"]  == target_lat_bin) & (bin_dict["lons"]  == target_lon_bin)
    pixels_within = slstr_da.where(xr.DataArray(mask, coords=slstr_da.coords), drop=True)

    return pixels_within


def compare_temperatures(soil_temp, veg_temp, TSURF, TSURFadj = None, MPDI =None, KUKA = None):
    """
    Gets the underlying SLSTR array of pixels for every AMSR2 Ka-LST pixel. Then calculates the mean and std for these, and plots
    """
    soil_array =  [] # Soil SLSTR pixels within AMSR2 pixel
    veg_array =  [] # Veg. SLSTR pixels within AMSR2 pixel

    veg_mean_list = []  # Mean of soil SLSTR pixels within AMSR2 pixel
    veg_std_list = [] # std of veg. SLSTR pixels within AMSR2 pixel

    soil_mean_list = []
    soil_std_list = []

    TSURF_list = []
    TSURFadj_list = []
    MPDI_list = []
    KUKA_list = []

    bin_dict = binning_smaller_pixels(soil_temp, TSURF)  # instead of soil_temp, any shoudl be good thats a SLSTR obs

    for targetlat in range(0, bin_dict["lats"].max()):
        for targetlon in range(0, bin_dict["lons"].max()):

            soil_subset = slstr_pixels_in_amsr2(soil_temp,
                                                bin_dict,
                                                targetlat,
                                                targetlon)

            veg_subset = slstr_pixels_in_amsr2(veg_temp,
                                               bin_dict,
                                               targetlat,
                                               targetlon)

            soil_array.append(subset_statistics(soil_subset)[0])
            veg_array.append(subset_statistics(veg_subset)[0])

            soil_mean_list.append(subset_statistics(soil_subset)[1]["mean"])
            soil_std_list.append(subset_statistics(soil_subset)[1]["std"])

            veg_mean_list.append(subset_statistics(veg_subset)[1]["mean"])
            veg_std_list.append(subset_statistics(veg_subset)[1]["std"])

            TSURF_subset = TSURF.isel(lat=targetlat, lon=targetlon)
            TSURF_list.append(TSURF_subset.values.item())

            if MPDI is not None:
                try:
                    MPDI_subset = MPDI.isel(lat=targetlat, lon=targetlon)
                    MPDI_list.append(MPDI_subset.values.item())

                    KUKA_subset = KUKA.isel(lat=targetlat, lon=targetlon)
                    KUKA_list.append(KUKA_subset.values.item())

                    TSURFadj_subset = TSURFadj.isel(lat=targetlat, lon=targetlon)
                    TSURFadj_list.append(TSURFadj_subset.values.item())

                except Exception as e:
                    print(e)

    df =  pd.DataFrame({
        "veg_temp": veg_mean_list,
        "veg_std": veg_std_list,
        "soil_temp": soil_mean_list,
        "soil_std": soil_std_list,
        "tsurf_ka": TSURF_list,
        "tsurf_adj": TSURFadj_list,
        "mpdi": MPDI_list,
        "kuka": KUKA_list,
        "soil_array": soil_array,
        "veg_array": veg_array,
    })

    return df

