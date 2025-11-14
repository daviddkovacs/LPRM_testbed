from readers.Sat import BTData, LPRMData
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

import pandas as pd
from utilities.utils import (bbox,
                             soil_canopy_temperatures,
                             mpdi,
                             extreme_hull_vals,
                             find_common_coords,
                             interceptor,
                             dummy_line)

from utilities.plotting import scatter_density,plot_maps
from config.paths import path_lprm, path_bt

list_bbox= [
    -122.5836129812065,
    42.225601574757036,
    -88.31189937769449,
    46.976817244415855
  ]

# Frequencies(AMSR2):
AMSR2_bands = ['6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0']
_path_bt = path_bt
_path_lprm = path_lprm
sat_freq = '18.7'
sat_sensor = "amsr2"
overpass = "day"
target_res = "10"

composite_start = "2024-10-01"
composite_end = "2024-10-01"

datelist = pd.date_range(start=composite_start, end=composite_end, freq="D")
datelist = [s.strftime("%Y-%m-%d") for s in datelist]


for d in datelist:

    BT = BTData(path = _path_bt,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    BT = BT.to_pandas()
    BT = bbox(BT, list_bbox)

    BT["MPDI"] =  mpdi(BT["BT_V"], BT["BT_H"])


    LPRM = LPRMData(path =_path_lprm,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    ds_LPRM = LPRM.to_xarray(bbox=list_bbox)
    ds_LPRM = ds_LPRM.assign_coords({"lon": ds_LPRM["LON"],
                                     "lat": ds_LPRM["LAT"]})

    LPRM = LPRM.to_pandas()
    LPRM = bbox(LPRM, list_bbox)

    print(f"{d} read")

    plt.ion()
    common_data = find_common_coords(LPRM,BT)

    x_var = "VOD_KU"
    y_var = "TSURF"
    x = common_data[x_var]
    y = common_data[y_var]

    scatter_density(
        ref=x,
        test=y,
        test_colour=common_data["SM_C1"],
        xlabel= x_var,
        ylabel=y_var,
        cbar_label= "SM_C1",
        # cbar_type = "jet",
        xlim = (0,1.4),
        ylim = (273,330),
        # cbar_scale = (0,0.5),
        # dpi =5
        )

    point_cloud = common_data.dropna(how = "any")

    _x = point_cloud[x_var]
    _y = point_cloud[y_var]

    points = np.array([_x,_y]).T
    hull = ConvexHull(points)

    vertices = extreme_hull_vals(points[hull.vertices, 0],
                                 points[hull.vertices, 1],
                                 x_variable= x_var,
                                 y_variable= y_var, )
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)

    # Gradient of warm edge (y2-y1) / (x2-x1)
    # 0th is x and 1st index is y coord
    grad_warm_edge = ((vertices[f"max_{y_var}"][1] - vertices[f"max_{x_var}"][1]) /
                 (vertices[f"max_{y_var}"][0] - vertices[f"max_{x_var}"][0]))

    # Intercept of warm edge on y-axis
    intercept_warm_edge = ((grad_warm_edge * vertices[f"max_{x_var}"][0]) * -1) + vertices[f"max_{x_var}"][1]
    plt.plot(x, grad_warm_edge * x + intercept_warm_edge, label = "Warm edge")

    # Cold edge
    cold_edge = vertices[f"min_{y_var}"][1]
    plt.axhline(cold_edge)

    # full vegetation cover edge
    full_veg_cover = vertices[f"max_{x_var}"][0]
    plt.axvline(full_veg_cover)

    temperatures_data = soil_canopy_temperatures(_x,
                                                _y,
                                                cold_edge,
                                                grad_warm_edge,
                                                intercept_warm_edge,
                                                full_veg_cover
                                                )

    point_cloud["T_SOIL"] = temperatures_data["T_soil_extreme"]
    point_cloud["T_CANOPY"] = temperatures_data["T_canopy_extreme"]

    # Grad and intercept for ALL points!
    point_cloud["gradient"] = temperatures_data["gradient_of_point"].values
    point_cloud["intercept"] = temperatures_data["intercept_of_point"].values

    point_cloud["p_o"], point_cloud["p_5"] = dummy_line(
        point_cloud["gradient"],point_cloud["intercept"])

    hull_x = points[hull.vertices, 0]
    hull_y = points[hull.vertices, 1]
    poly = Polygon((x, y) for x, y in zip(hull_x, hull_y))

    results = list(map(lambda p: interceptor(poly=poly, p_0 = p[0], p_5 = p[1], TSURF =p[2]),
                       zip(point_cloud["p_o"], point_cloud["p_5"], point_cloud["TSURF"])))

    point_cloud["T_soil_hull"], point_cloud["T_canopy_hull"] = zip(*results)

    cbar_lut = {"T_SOIL": (270, 330),
                "T_soil_hull": (270, 330),
                "TSURF": (270, 330),
                "T_canopy_hull": (270, 330),
                }

    plot_maps(point_cloud, cbar_lut)