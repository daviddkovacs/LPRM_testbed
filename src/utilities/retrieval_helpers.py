import os.path
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from shapely.geometry import LineString,  Point
from lprm.retrieval.lprm_v6_1.run_lprmv6 import load_band_from_ds
import numpy as np
import rioxarray
import pandas as pd
from shapely.geometry.multilinestring import MultiLineString

from utilities.utils import bbox

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

    t_dict = {"gradient_of_point" : gradient_of_point,
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
    if (isinstance(intersection, LineString) or isinstance(intersection, MultiLineString)) and not intersection.is_empty:
        t_soil, t_canopy = [list(intersection.coords)[i][1] for i in range(0,2)]
    if isinstance(intersection, Point):
        t_soil = t_canopy = list(intersection.coords)[0][1]
    if intersection.is_empty:
        t_soil = t_canopy = TSURF

    return t_soil, t_canopy


def get_aux(path,var):

    dataset = rioxarray.open_rasterio(path)[0,:,:]
    dataset = dataset.assign_coords(
        x=((dataset.x + 180) % 360) - 180,
        y=((dataset.y + 90) % 180) - 90,
    )
    dataarray = dataset.rename({"x" : "LON",
                                "y" : "LAT"})
    dataarray = dataarray.drop_vars(["band", "spatial_ref"])
    panda = dataarray.to_dataframe(var).reset_index()

    return panda


def tiff_df(path,lista = (-180,-90,90,180),target_res = "25"):

    path_BLD = os.path.join(path,f"auxiliary_data_BLD_{target_res}km")
    path_SND = os.path.join(path,f"auxiliary_data_SND_{target_res}km")
    path_CLY = os.path.join(path,f"auxiliary_data_CLY_{target_res}km")

    BLD = get_aux(path_BLD, "BLD")
    SND = get_aux(path_SND, "SND")
    CLY = get_aux(path_CLY, "CLY")

    _common_df = pd.merge(BLD,SND,
                         how='inner', on = ["LON","LAT"],)
    common_df = pd.merge(_common_df,CLY,
                         how='inner', on = ["LON","LAT"],)
    common_df = common_df.set_index(["LAT","LON"])
    subset_panda = bbox(common_df,lista)

    return subset_panda


def retrieve_LPRM(common_data,
                  aux_df,
                  sat_sensor,
                  sat_band,
                  T_soil_test = None,
                  T_canopy_test = None,
                  ):

    # Retrieve LPRM here

    merged = common_data.join(aux_df, how="inner")

    specs = get_specs(sat_sensor.upper())
    params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)
    freq = get_specs(sat_sensor.upper()).frequencies[sat_band.upper()]
    if "time" in merged.index.names:
        merged = merged.reset_index(level="time", drop=True)
    merged_geo = merged.to_xarray()

    if T_soil_test and T_canopy_test:
        T_soil_test = np.broadcast_to(T_soil_test, merged_geo["BT_V"].values.shape).astype('double')
        T_canopy_test = np.broadcast_to(T_canopy_test, merged_geo["BT_V"].values.shape).astype('double')

    sm, vod = par100.run_band(
        merged_geo["BT_V"].values.astype('double'),
        merged_geo["BT_H"].values.astype('double'),
        merged_geo["TSURF"].values.astype('double'),
        merged_geo["SND"].values.astype('double'),
        merged_geo["CLY"].values.astype('double'),
        merged_geo["BLD"].values.astype('double'),
        params.Q,
        params.w,
        params.opt_atm,
        specs.incidence_angle[0],
        params.h1,
        params.h2,
        params.vod_Av,
        params.vod_Bv,
        float(freq),
        params.temp_freeze,
        False,
        None,
        T_soil =  T_soil_test if T_soil_test else merged_geo["T_soil_hull"].values.astype('double'),
        T_canopy = T_canopy_test if T_canopy_test else merged_geo["T_canopy_hull"].values.astype('double'),
    )

    merged_geo[f"SM_ADJ"] = (("LAT", "LON"), sm)
    merged_geo[f"SM_ADJ"] = merged_geo[f"SM_ADJ"].where(merged_geo[f"SM_ADJ"] != -2, np.nan)

    merged_geo[f"VOD_ADJ"] = (("LAT", "LON"), vod)
    merged_geo[f"VOD_ADJ"] = merged_geo[f"VOD_ADJ"].where(merged_geo[f"VOD_ADJ"] != -2, np.nan)

    # merged_geo[f"DIF_SM{sat_band}-ADJ"] = merged_geo[f"SM_ADJ"] - merged_geo[f"SM_{sat_band}"]

    return  merged_geo


