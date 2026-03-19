from typing import Literal
import numpy as np
import xarray as xr

frequencies = {'C1': 6.9, 'C2': 7.3, 'X': 10.7, 'KU': 18.7, 'K': 23.8, 'KA': 36.5}

landcover_bbox_lut = {
    "desert":
        [
            -108.63606294851671,
            35.737694493846305,
            -107.15556437732934,
            36.378059550297394
        ]
    ,
    "cropland":
        [
            -106.19766529481865,
            37.21339806346914,
            -105.69751210757258,
            37.712219835567026
        ]
    ,
    "forest":
        [
            -105.51860007675909,
            36.41386690513728,
            -104.52136945543491,
            37.479346477831314
        ]
    ,
    "forest_amazon":
[
    -73.14282831871029,
    -10.252603756879822,
    -69.79762012909454,
    -7.40706163002595
  ]
    ,
    "forest_deciduous":
[
    22.517240470154462,
    48.47465603576933,
    23.298305514714286,
    49.06651088674198
  ]
    ,
    "desert_pixel":
[
    -108.34199700561919,
    35.727924643500515,
    -107.92392062121272,
    35.95425540038144
  ]
    ,
    "forest_west_virginia":
        [
            -82.81946874253154,
            37.06066523546711,
            -81.4855328438654,
            37.886633912495896
        ]
    ,
    "agri_mix":
[
    -106.07808475806138,
    36.382928089727386,
    -105.08140915589864,
    37.443518841830155
  ]
}

def calc_Holmes_temp(AMSR2):
    """
    Surface temperature from Ka-band observations according to Holmes et al. 2008
    """
    TSURF = AMSR2["bt_36.5V"] * 0.893 + 44.8
    TSURF.attrs = AMSR2.attrs
    return TSURF


def MW_fraction(AMSR2, num = "ku", denom = "ka"):
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


def threshold_ndvi(lst, ndvi, soil_range=[0,0.2], veg_range=[0.5,1]):
    """
    Simple thresholding of Soil-Veg to get different temps.
    """

    veg_mask = (ndvi > veg_range[0]) & (ndvi <= veg_range[1])
    veg_temp = lst.where(veg_mask)

    soil_mask = (ndvi >= soil_range[0]) & (ndvi < soil_range[1])
    soil_temp = lst.where(soil_mask)

    return soil_temp, veg_temp


def crop2roi(ds,bbox):
    """
    Cropping to bbox. Handles S3 projection (lat, lon) for each coord
    :param ds:
    :param bbox:
    :return:
    """
    valid = (
            (ds.lon >= bbox[0]) & (ds.lon <= bbox[2]) &
            (ds.lat >= bbox[1]) & (ds.lat <= bbox[3])
    )
    return ds.where(valid, drop=True)

def ravel_roi_time(data, roi ,time, method = None):
    """
    Selects data under region of interest, and gets time coord nearest to obs.
    :param data: Dataarray
    :param roi: bbox
    :param time:
    :return: numpy vector
    """

    return crop2roi(data, roi).sel(time=time, method=method).values.ravel()


def filternan(array):
    return  array.values.flatten()[~np.isnan(array.values.flatten())]


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
    This is the bread-and-butter of the high-low resolution matching.
    Linear indexing is used for speed.
    Understanding how binning works: users please visit: https://numpy.org/doc/2.3/reference/generated/numpy.digitize.html
    We still have outlier pixels from MODIS, has it is an irregular swath data, and these cant be dropped (to withhold nxm structure)
    This should work on temporal data cubes (3d)
    :param highres_da: SLSTR/MODIS da
    :param lowres_da: AMSR2 data
    :return: coarsened MODIS to AMSR2 footprint
    """

    bin_dict = binning_smaller_pixels(highres_da, lowres_da)

    n_time = bin_dict["lats"].shape[0]
    n_lat = lowres_da.sizes['lat']
    n_lon = lowres_da.sizes['lon']
    spatial_area = n_lat * n_lon

    lats = bin_dict["lats"].flatten()
    lons = bin_dict["lons"].flatten()

    t_indices = np.arange(n_time)
    time_mask = np.broadcast_to(t_indices[:, None, None], bin_dict["lats"].shape).flatten()

    # We do not need bins that are 0--> outside of bbox
    valid_mask = (lats > 0) & (lats <= n_lat) & (lons > 0) & (lons <= n_lon)

    # vectorized bins, each pixel gets one according to where it is in time:
    # e.g.: bin #3 will not be bin #3 on two diff. days, it might be e.g.: 18 and 34 depending on
    # how much data there is.
    combined_bins = (time_mask * spatial_area) + (lats * n_lon) + lons

    # We stack the MODIS data to have one dim. encapsulating all original ones.
    hr_stacked = highres_da.stack(pixel=("time", "row", "column"))

    # We only care about pixels that actually fell inside the lowres grid
    # Setting invalid bins to -1 so we can filter them
    final_bins = np.where(valid_mask, combined_bins, -1)
    hr_stacked.coords["bin_id"] = ("pixel", final_bins)

    # Groupby and Mean (ignoring the -1 'out of bounds' bin)
    stats = hr_stacked.where(hr_stacked.bin_id >= 0, drop=True).groupby("bin_id").mean().compute()

    # Map results to the flat output array (size 44 * 5 * 6 = 1320)
    output_flat = np.full(n_time * spatial_area, np.nan)

    # Fill the flat array using the indices we just calculated
    _indices = stats.bin_id.values.astype(int)
    indices = _indices - _indices.min()
    output_flat[indices] = stats.values

    # Reshape to 3D: (Time, Lat, Lon)
    reshaped_data = output_flat.reshape((n_time, n_lat, n_lon))

    # Create final DataArray
    bin_da = xr.DataArray(
        data=reshaped_data,
        coords={
            "time": highres_da.time.values,
            "lat": lowres_da.lat.values,
            "lon": lowres_da.lon.values
        },
        dims=("time", "lat", "lon")
    )

    return bin_da


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


def morning_evening_passes(dataset,
                           time_of_day:Literal["morning","evening"],
                           threshold = 14):
    """
    We are in Central Standard Time (US Midwest)--> UTC - 6
     e.g.: 20:00 UTC is 14:00 over the Midwest, and an ascending overpass for A-train since crossing time is 13:30
    :param dataset: dataset to split (NDVI is only daytime!!!!)
    :param threshold: Upon which hour the split happens
    :return: day and nighttime datasets
    """
    if time_of_day == "morning":
        is_ndvi = isinstance(dataset, xr.DataArray) and "ndvi" in dataset.name.lower()

        if is_ndvi:
            _dataset = xr.full_like(dataset, fill_value=np.nan)
        else:
            _dataset = dataset.where(dataset.time.dt.hour <= threshold, drop=True)

    elif time_of_day == "evening":
        _dataset = dataset.where(dataset.time.dt.hour >= threshold, drop=True)
    else:
        raise Exception("Please select time of day: 'morning' or 'evening'")

    return _dataset


def common_observations(refds, ds2, method = "nearest"):
    if method == "exact":
        method = None
    if len(refds.time) > len(ds2.time):
        raise Exception("refds should have LESS observations, than ds2. Switch the two args!")

    ds2_reduced = ds2.sel(time=refds.time, method=method)

    return refds, ds2_reduced


def match_MYD09_to_MYD11(LST_full,NDVI_full):

    common_dates = np.intersect1d(LST_full.time.values,NDVI_full.time.values)
    LST, NDVI = LST_full.sel(time= common_dates), NDVI_full.sel(time =common_dates)

    # Unfortunately we will have to stack to vectors here, as finding common lat/lons in 3D is hard
    LST_flat = LST.stack(obs=  ("time","row","column"))
    NDVI_flat = NDVI.stack(obs=  ("time","row","column"))

    # okay so:
    # The issue with finding the intersection in lat, and lon only is NOT possible
    # becuase we are not dealing with a fixed grid, but swaths. Every pixel has a (lat,lon) pair
    # we use a complex notation, to combine the coords find the common pixels
    lst_pairs = LST_flat.lat.values + 1j * LST_flat.lon.values
    ndvi_pairs = NDVI_flat.lat.values + 1j * NDVI_flat.lon.values

    common_pixels= np.intersect1d(lst_pairs, ndvi_pairs)

    lst_mask = np.isin(lst_pairs, common_pixels)
    ndvi_mask = np.isin(ndvi_pairs, common_pixels)

    LST_filtered = LST_flat.isel(obs=lst_mask)
    NDVI_filtered = NDVI_flat.isel(obs=ndvi_mask)

    LST_unstacked = LST_filtered.unstack("obs").drop_vars(["row", "column"])
    NDVI_unstacked = NDVI_filtered.unstack("obs").drop_vars(["row", "column"])

    return LST_unstacked, NDVI_unstacked
