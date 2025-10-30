import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd
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