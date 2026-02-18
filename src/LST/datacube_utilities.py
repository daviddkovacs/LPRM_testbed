import numpy as np
import pandas as pd
import xarray as xr

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



def binning_smaller_pixels(high_res, low_res):
    res = low_res.attrs["resolution"]
    lat_edges = get_edges(low_res.lat.values, res)
    lon_edges = get_edges(low_res.lon.values, res)

    iterables = {}

    iterables["lats"] = np.digitize(high_res.lat.values, lat_edges)
    iterables["lons"] = np.digitize(high_res.lon.values, lon_edges)

    return iterables

def coarsen_highres(highres_da, lowres_da):
    """
    We perfom the binning of the MODIS/SLSTR pixels to the coarse AMSR2 resolution.
    Linear indexing is used for speed.
    :param highres_da: SLSTR/MODIS da
    :param lowres_da: AMSR2 data
    :return:
    """

    bin_dict = binning_smaller_pixels(highres_da, lowres_da)

    n_time = bin_dict["lats"].shape[0]  # 44
    n_lat = lowres_da.sizes['lat']  # 5
    n_lon = lowres_da.sizes['lon']  # 6
    spatial_area = n_lat * n_lon

    # 2. Extract indices and flatten them
    lats = bin_dict["lats"].flatten()
    lons = bin_dict["lons"].flatten()

    # Create the time component for every high-res pixel
    t_indices = np.arange(n_time)
    time_mask = np.broadcast_to(t_indices[:, None, None], bin_dict["lats"].shape).flatten()

    # 3. CRITICAL: Filter out indices that are out of bounds
    # This prevents 'Index 1320 is out of bounds'
    valid_mask = (lats >= 0) & (lats < n_lat) & (lons >= 0) & (lons < n_lon)

    # 4. Calculate 3D Linear ID: (t * spatial_area) + (lat * n_lon) + lon
    combined_bins = (time_mask * spatial_area) + (lats * n_lon) + lons

    # 5. Stack highres data and assign the IDs
    hr_stacked = highres_da.stack(pixel=("time", "row", "column"))

    # We only care about pixels that actually fell inside the lowres grid
    # Setting invalid bins to -1 so we can filter them
    final_bins = np.where(valid_mask, combined_bins, -1)
    hr_stacked.coords["bin_id"] = ("pixel", final_bins)

    # 6. Groupby and Mean (ignoring the -1 'out of bounds' bin)
    stats = hr_stacked.where(hr_stacked.bin_id >= 0, drop=True).groupby("bin_id").mean().compute()

    # 7. Map results to the flat output array (size 44 * 5 * 6 = 1320)
    output_flat = np.full(n_time * spatial_area, np.nan)

    # Fill the flat array using the indices we just calculated
    # stats.bin_id contains the 'address' in the 1320-length array
    indices = stats.bin_id.values.astype(int)
    output_flat[indices] = stats.values

    # 8. Reshape to 3D: (Time, Lat, Lon)
    reshaped_data = output_flat.reshape((n_time, n_lat, n_lon))

    # 9. Create final DataArray
    bin_da = xr.DataArray(
        data=reshaped_data,
        coords={
            "time": highres_da.time.values,
            "lat": lowres_da.lat.values,
            "lon": lowres_da.lon.values
        },
        dims=("time", "lat", "lon")
    )

    plotdate = "2018-02-04T08:43:13"

    return bin_da

# def highres_pixels_in_lowres(highres_da,
#                           bin_dict,
#                           target_lat_bin,
#                           target_lon_bin,):
#     mask_3d = (bin_dict["lats"]  == target_lat_bin) & (bin_dict["lons"]  == target_lon_bin)
#     mask_da = xr.DataArray(mask_3d, dims=["time", "row", "column"])
#     pixels_within = highres_da.where(mask_da, drop=False)
#     return pixels_within.mean()


# def compare_temperatures(soil_temp, veg_temp, TSURF, TSURFadj = None, MPDI =None, KUKA = None):
#     """
#     Gets the underlying SLSTR array of pixels for every AMSR2 Ka-LST pixel. Then calculates the mean and std for these, and plots
#     """
#     soil_array =  [] # Soil SLSTR pixels within AMSR2 pixel
#     veg_array =  [] # Veg. SLSTR pixels within AMSR2 pixel
#
#     veg_mean_list = []  # Mean of soil SLSTR pixels within AMSR2 pixel
#     veg_std_list = [] # std of veg. SLSTR pixels within AMSR2 pixel
#
#     soil_mean_list = []
#     soil_std_list = []
#
#     TSURF_list = []
#     TSURFadj_list = []
#     MPDI_list = []
#     KUKA_list = []
#
#     bin_dict = binning_smaller_pixels(soil_temp, TSURF)  # instead of soil_temp, any shoudl be good thats a SLSTR obs
#
#     for targetlat in range(0, bin_dict["lats"].max()):
#         for targetlon in range(0, bin_dict["lons"].max()):
#
#             soil_subset = highres_pixels_in_lowres(soil_temp, bin_dict, targetlat, targetlon)
#
#             veg_subset = highres_pixels_in_lowres(veg_temp, bin_dict, targetlat, targetlon)
#
#             soil_array.append(subset_statistics(soil_subset)[0])
#             veg_array.append(subset_statistics(veg_subset)[0])
#
#             soil_mean_list.append(subset_statistics(soil_subset)[1]["mean"])
#             soil_std_list.append(subset_statistics(soil_subset)[1]["std"])
#
#             veg_mean_list.append(subset_statistics(veg_subset)[1]["mean"])
#             veg_std_list.append(subset_statistics(veg_subset)[1]["std"])
#
#             TSURF_subset = TSURF.isel(lat=targetlat, lon=targetlon)
#             TSURF_list.append(TSURF_subset.values.item())
#
#             if MPDI is not None:
#                 try:
#                     MPDI_subset = MPDI.isel(lat=targetlat, lon=targetlon)
#                     MPDI_list.append(MPDI_subset.values.item())
#
#                     KUKA_subset = KUKA.isel(lat=targetlat, lon=targetlon)
#                     KUKA_list.append(KUKA_subset.values.item())
#
#                     TSURFadj_subset = TSURFadj.isel(lat=targetlat, lon=targetlon)
#                     TSURFadj_list.append(TSURFadj_subset.values.item())
#
#                 except Exception as e:
#                     print(e)
#
#     df =  pd.DataFrame({
#         "veg_temp": veg_mean_list,
#         "veg_std": veg_std_list,
#         "soil_temp": soil_mean_list,
#         "soil_std": soil_std_list,
#         "tsurf_ka": TSURF_list,
#         "tsurf_adj": TSURFadj_list,
#         "mpdi": MPDI_list,
#         "kuka": KUKA_list,
#         "soil_array": soil_array,
#         "veg_array": veg_array,
#     })
#
#     return df


def clean_pad_data(list_of_da, x , y):
    """
    filters empy cropped data from SLSTR/MODIS as well as pads data so each cropped ROI is the same dm
    :param list_of_da: list of dataarrays to be concatenated.
    :param x: Variable at x diension--> SLSTR: "rows", MODIS : "lon"
    :param y: Variable at y diension--> SLSTR: "columns", MODIS : "lat"

    :return: list of nicely cropped dataarrays
    """
    cleaned_data = [
        ds for ds in list_of_da
        if ds.sizes[x] > 0 and ds.sizes[y] > 0
    ]
    max_x = max(ds.sizes[x] for ds in cleaned_data)
    max_y = max(ds.sizes[y] for ds in cleaned_data)

    padded_data = []

    for ds in cleaned_data:
        pad_x = max_x - ds.sizes[x]
        pad_y = max_y - ds.sizes[y]

        pad_kwargs = {
            x: (0, pad_x),
            y: (0, pad_y),
            "constant_values": np.nan
        }

        ds_padded = ds.pad(**pad_kwargs)
        padded_data.append(ds_padded)

    return padded_data

def morning_evening_passes(dataset, threshold = 14):
    """
    We are in Central Standard Time (US Midwest)--> UTC - 6
     e.g.: 20:00 UTC is 14:00 over the Midwest, and an ascending overpass for A-train since crossing time is 13:30
    :param dataset: dataset to split (NDVI is only daytime!!!!)
    :param threshold: Upon which hour the split happens
    :return: day and nighttime datasets
    """
    morning_dataset  = dataset.where(dataset.time.dt.hour <= threshold, drop=True)
    afternoon_dataset = dataset.where(dataset.time.dt.hour >= threshold, drop=True)
    return morning_dataset, afternoon_dataset

def common_observations(refds, ds2):
    if len(refds.time) > len(ds2.time):
        raise Exception("refds should have LESS observations, than ds2. Switch the two args!")

    ds2_reduced = ds2.sel(time=refds.time, method='nearest')

    return refds, ds2_reduced