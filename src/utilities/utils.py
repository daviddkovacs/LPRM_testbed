import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd
from scipy.spatial import ConvexHull
import math
from pandas import Timedelta
pd.options.mode.chained_assignment = None



def to_radians(df,
               lat = "lat",
               lon = "lon"):
    """

    Parameters
    ----------
    df: pandas dataframe with lat, lon columns as degrees
    lat: name of the latitude column. Default is 'lat'
    lon: name of the longitude column. Default is 'lon'

    Returns
    -------
    df: pandas dataframe with lat, lon columns as radians
    """

    df.loc[:, f"rad_{lat}"] = np.deg2rad(df[lat].values)
    df.loc[:, f"rad_{lon}"] = np.deg2rad(df[lon].values)

    return df


def filter_distance(distance_df,
                    radius = 10):
    """

    Parameters
    ----------
    distance_df: pandas dataframe with nearest distances and ids
    radius: threshold radius for nearest distances. Default is 10

    Returns
    -------
    radius_df: pandas dataframe with nearest distances filtered
    """

    radius_df = distance_df[distance_df["distance"]<radius]

    return radius_df


def nn_loc_search(ref_obj,
                  test_obj,
                  ):
    """

    Parameters
    ----------
    test_obj: dataframe to construct BallTree with: the dataset to "search" from
    ref_obj: dataframe to query: the dataset to "reference" from

    Ball Tree NN search (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)

    df2 will be the query/reference points, the coords in df1 will be searched against it
    Only works with radians! using the Haversine formula to calculate the distance.

    Returns
    -------
    df_dist_ids: df containing nn dist between df1 and df2, the ids where this happens in df1, and df2 original ids
    """

    ref_obj = to_radians(ref_obj)
    test_obj = to_radians(test_obj)

    ball = BallTree(test_obj[["rad_lat", "rad_lon"]].values, metric='haversine')
    distances, indices_nn = ball.query(ref_obj[["rad_lat", "rad_lon"]].values, k=1)

    radius_earth = 6371 # km
    distances = distances.ravel() * radius_earth
    indices_nn = indices_nn.ravel()
    index_ref = ref_obj.index

    df_dist_ids = pd.DataFrame({"distance": distances,"index_nn": indices_nn, "index_ref": index_ref})

    return df_dist_ids


def collocate_datasets(ref_obj,
                       test_obj,
                       *args
                    ):

    # Find NN observations to all locs in air_data
    nearest_locs = nn_loc_search(ref_obj, test_obj)
    # Filter NN to a max radius, i.e.: 15km
    filtered_nearest_locs = filter_distance(nearest_locs)

    # Filter both datasets to NN<15km observations
    ref_nn = ref_obj.iloc[filtered_nearest_locs["index_ref"]].reset_index(drop=True)
    test_nn = test_obj.iloc[filtered_nearest_locs["index_nn"]].reset_index(drop=True)

    ref_nn = ref_nn.drop(columns = ["rad_lon","rad_lat"])
    test_nn = test_nn.drop(columns = ["rad_lon","rad_lat"])

    return ref_nn, test_nn


def mpdi(v_freq,
         h_freq):
    """

    Parameters
    ----------
    v_freq: vertical polarized brightness temperature
    h_freq: horizontal polarized brightness temperature

    Returns
    -------
    mpdi: Microwave Polarisation Difference Index
    """

    mpdi = (v_freq-h_freq) / (v_freq + h_freq)

    return mpdi

def bbox(df,
         lista):

    # df = df.loc[df["LAT"] > lista[1]]
    # df = df.loc[df["LAT"] < lista[3]]
    #
    # df = df.loc[df["LON"] > lista[0]]
    # df = df.loc[df["LON"] < lista[2]]

    df = df.loc[df.index.get_level_values("LAT") > lista[1]]
    df = df.loc[df.index.get_level_values("LAT") < lista[3]]

    df = df.loc[df.index.get_level_values("LON") > lista[0]]
    df = df.loc[df.index.get_level_values("LON") < lista[2]]

    return df

def calc_surface_temperature(bt_Ka_input: np.ndarray) -> np.ndarray:
    """
    Calculates surface temperature as needed for LPRM radiative transfer equation.

    Parameters
    ----------
    bt_Ka_input (np.ndarray): Ka-band BT input, to retrieve surface temperature. Usually measured by sensor (not SMAP).

    Returns
    -------
    temperature (np.ndarray): Surface temperature in Kelvins.
    """

    temperature = (0.893 * bt_Ka_input) + 44.8
    # temperature = np.where(temperature <= 274.15, np.nan, temperature)

    return temperature


def find_common_coords(df1,
                       df2,
                       target_res,
                       dropna_subset= ['BT_V',"TSURF"],
                       suffixes = ('_LPRM', '_BT')):
    """
    Finding common locs between two datasets. Needed to avoid dimension mismatch when plotting.

    :df1 ref: dataframe reference
    :df2 test:  dataframe test (must add offset to match ref coords)
    :return: common dataframe
    """
    offset_lut = {"10" : 0.05,
                  "25" : 0.125}

    # bt = bt.drop(columns = ["SCANTIME"])
    # lprm = lprm.drop(columns = ["FLAGS"])
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df2["LAT"] = df2["LAT"] + offset_lut[target_res]
    df2["LON"] = df2["LON"] + offset_lut[target_res]

    df1['LAT'] = df1['LAT'].round(4)
    df1['LON'] = df1['LON'].round(4)
    df2['LAT'] = df2['LAT'].round(4)
    df2['LON'] = df2['LON'].round(4)

    common_df = pd.merge(df1, df2 ,how='inner', on = ["LON","LAT"], suffixes=suffixes)
    common_df =  common_df.dropna(subset=dropna_subset)

    cordinates = [common_df["LAT"].values, common_df["LON"].values]
    mi_array = zip(*cordinates)
    common_df.index = pd.MultiIndex.from_tuples(mi_array, names=["LAT", "LON"])

    return common_df

def normalize(array):
    """
    normalize data between [0:1] according to Carlson (2020) "A Brief Analysis.."
    """

    minimum = array.min()
    maximum = array.max()

    scaled_array = (array - minimum) / (maximum -minimum )
    scaled_array = np.positive(scaled_array)
    return scaled_array


def extreme_hull_vals(x_values,
                      y_values,
                      x_variable = "VOD_KU",
                      y_variable = "TSURF"):


    hull_df = pd.DataFrame({
        x_variable: x_values,
        y_variable: y_values
    })

    min_x = hull_df.loc[hull_df.loc[hull_df[x_variable] == hull_df[x_variable].min(), y_variable].idxmin()].values
    max_x = hull_df.loc[hull_df.loc[hull_df[x_variable] == hull_df[x_variable].max(), y_variable].idxmin()].values

    min_y = hull_df.loc[hull_df.loc[hull_df[y_variable] == hull_df[y_variable].min(), x_variable].idxmin()].values
    max_y = hull_df.loc[hull_df.loc[hull_df[y_variable] == hull_df[y_variable].max(), x_variable].idxmin()].values

    vertex_dict = {
        f"min_{x_variable}" : min_x,
        f"max_{x_variable}" : max_x,
        f"min_{y_variable}" : min_y,
        f"max_{y_variable}" : max_y,
                  }

    return vertex_dict


def get_dates(composite_start,composite_end, freq = "ME"):

    datelist = pd.date_range(start=composite_start, end=composite_end, freq=freq)

    return datelist

def save_nc(ds,path):
    """
    saving datasets with converting time dim from "object" type, as it raised errors.
    :param ds: input dataset
    :param path: full path with filename and extension (.nc)
    """
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in ds.data_vars}
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
    ds.to_netcdf(path, encoding =  encoding)

def convex_hull(points):

    hull = ConvexHull(points)

    hull_x = points[hull.vertices, 0]
    hull_y = points[hull.vertices, 1]

    # We add the "close to last" coord, so that the hull definetely closes!!
    last_x , last_y = (points[hull.vertices, 0][0],
                       points[hull.vertices, 1][0])

    hull_x = np.append(hull_x,last_x)
    hull_y = np.append(hull_y,last_y)

    return hull_x, hull_y


def pearson_corr(da1,
                 column1,
                 da2,
                 column2
                 ):

    df_1 = da1.to_dataframe()
    df_2 = da2.to_dataframe()
    df_m = pd.concat([df_1, df_2], axis=1)
    print(df_m)
    r = df_m.corr(method="pearson").loc[column1, column2]

    return np.round(r,2)


def local_solar_time(scantime,
            DOY,
            long):
    # use this for the dataframe
    site_UTC = DOY + Timedelta(seconds=scantime)
    site_solar = site_UTC + Timedelta(hours= long/ (math.pi * 12))
    return site_solar
