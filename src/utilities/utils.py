import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd
pd.options.mode.chained_assignment = None
from shapely.geometry import LineString,  Point

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
         list):

    df = df.loc[df["LAT"] > list[1]]
    df = df.loc[df["LAT"] < list[3]]

    df = df.loc[df["LON"] > list[0]]
    df = df.loc[df["LON"] < list[2]]

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


def find_common_coords(lprm, bt):
    """
    Finding common locs between two datasets. Needed to avoid dimension mismatch when plotting.

    :param ref: dataframe reference
    :param test:  dataframe test
    :return: common dataframe
    """
    # bt = bt.drop(columns = ["SCANTIME"])
    lprm = lprm.drop(columns = ["FLAGS"])
    bt["LAT"] = bt["LAT"] + 0.05
    bt["LON"] = bt["LON"] + 0.05

    lprm['LAT'] = lprm['LAT'].round(4)
    lprm['LON'] = lprm['LON'].round(4)
    bt['LAT'] = bt['LAT'].round(4)
    bt['LON'] = bt['LON'].round(4)

    common_df = pd.merge(lprm,bt,how='inner', on = ["LON","LAT"], suffixes=('_LPRM', '_BT'))
    common_df =  common_df.dropna(subset=['BT_V',"TSURF"])

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

def soil_canopy_temperatures(point_x,
                             point_y,
                             cold_edge,
                             grad_warm_edge,
                             intercept_warm_edge,
                             full_veg_cover
                             ):
    # equation of line: Tsoil = grad_warm_edge * point_x + intercept_warm_edge
    a = point_y - cold_edge
    b = (grad_warm_edge * point_x + intercept_warm_edge) - point_y

    A = cold_edge
    D = intercept_warm_edge
    T_soil_extreme = ((a / (a+b)) * (D - A) + A)

    B = cold_edge
    C = (grad_warm_edge * full_veg_cover + intercept_warm_edge)
    T_canopy_extreme = ((a / (a + b)) * (C -B ) + B)

    # these gradients and intercepts are needed to find the intersection for every line for every point with the hull
    # yes, I know I define multiple variables to be the same, but (for now) it is better to understand..
    gradient_of_point =  (( T_soil_extreme - T_canopy_extreme) / (0 - full_veg_cover))
    intercept_of_point = T_soil_extreme

    t_dict = {"T_soil_extreme" : T_soil_extreme,
              "T_canopy_extreme": T_canopy_extreme,
              "gradient_of_point" : gradient_of_point,
              "intercept_of_point" : intercept_of_point}

    return t_dict


def dummy_line(gradient, intercept):

    # We need to get two arbitrary points of the line
    # To find the intersection with the hull
    # y_0 = intercept
    p_5 = (gradient * 5) + intercept
    p_0 = intercept

    return p_0, p_5


def interceptor(poly, p_0, p_5, TSURF):

    line = LineString([(0,p_0) ,(5, p_5)])

    intersection = poly.intersection(line)
    if isinstance(intersection, LineString) and not intersection.is_empty:
        t_soil, t_canopy = [list(intersection.coords)[i][1] for i in range(0,2)]
    if isinstance(intersection, Point):
        t_soil = t_canopy = list(intersection.coords)[0][1]
    if intersection.is_empty:
        t_soil = t_canopy= TSURF

    return t_soil, t_canopy