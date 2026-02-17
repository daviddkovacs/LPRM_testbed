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
