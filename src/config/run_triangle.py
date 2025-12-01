import datetime
import pandas as pd
from scipy.stats import pearsonr
from readers.Sat import BTData, LPRMData
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import dask.array as da
from utilities.utils import (
    bbox,
    mpdi,
    extreme_hull_vals,
    find_common_coords,
    get_dates,
    convex_hull,
    pearson_corr
)
from utilities.retrieval_helpers import (
    soil_canopy_temperatures,
    interceptor,
    dummy_line, retrieve_LPRM,
)
from utilities.plotting import scatter_density, plot_maps_LPRM, plot_maps_day_night, plot_timeseries
from config.paths import path_lprm, path_bt, path_aux

list_bbox= [
    -168.63835264498513,
    -48.57856090133727,
    173.51106084596557,
    70.73328245211653
  ]

# Frequencies(AMSR2):
AMSR2_bands = ['6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0']
sat_band = 'X'
sat_sensor = "amsr2"
overpass = "night"
target_res = "25"

composite_start = "2024-06-01"
composite_end = "2024-06-01"

datelist = get_dates(composite_start, composite_end, freq = "D")

dt_original_ts = []
dt_adjusted_ts = []
nt_ts = []

for d in datelist:
    try:
        BT_object = BTData(path = path_bt,
                       date = d,
                       sat_freq = sat_band,
                       overpass = overpass,
                       sat_sensor = sat_sensor,
                       target_res = target_res,
                       )

        BT = BT_object.to_pandas()
        BT = bbox(BT, list_bbox)

        BT["MPDI"] =  mpdi(BT["BT_V"], BT["BT_H"])

        LPRM_object = LPRMData(path =path_lprm,
                       date = d,
                       sat_freq = sat_band,
                       overpass = overpass,
                       sat_sensor = sat_sensor,
                       target_res = target_res,
                       )

        LPRM = LPRM_object.to_pandas()
        LPRM = bbox(LPRM, list_bbox)

        night_LPRM_object = LPRMData(path =path_lprm,
                       date = d,
                       sat_freq = sat_band,
                       overpass = "night",
                       sat_sensor = sat_sensor,
                       target_res = target_res,
                       )

        # night_LPRM = night_LPRM_object.to_pandas()
        night_LPRM = night_LPRM_object.to_xarray(list_bbox)

        print(f"{d} read")

        plt.ion()
        common_data = find_common_coords(BT,LPRM,target_res)

        x_var = "VOD_KU"
        y_var = "TSURF"
        common_data = common_data.dropna(how = "any")

        x = common_data[x_var]
        y = common_data[y_var]

        scatter_density(
            ref=x,
            test=y,
            test_colour=common_data[f"SM_{sat_band}"],
            xlabel= x_var,
            ylabel=y_var,
            cbar_label= f"SM_{sat_band}",
            # cbar_type = "jet",
            xlim = (0,1.4),
            ylim = (273,330),
            cbar_scale = (0,0.5),
            # dpi =5
            )

        points = np.array([x,y]).T
        hull_x, hull_y = convex_hull(points)
        vertex = extreme_hull_vals(hull_x,
                                   hull_y,
                                   x_variable= x_var,
                                   y_variable= y_var, )

        # Gradient of warm edge (y2-y1) / (x2-x1)
        # 0th is x and 1st index is y coord
        grad_warm_edge = ((vertex[f"max_{y_var}"][1] - vertex[f"max_{x_var}"][1]) /
                     (vertex[f"max_{y_var}"][0] - vertex[f"max_{x_var}"][0]))

        # Intercept of warm edge on y-axis
        intercept_warm_edge = ((grad_warm_edge * vertex[f"max_{x_var}"][0]) * -1) + vertex[f"max_{x_var}"][1]

        # Cold edge
        cold_edge = vertex[f"min_{y_var}"][1]

        # full vegetation cover edge
        full_veg_cover = vertex[f"max_{x_var}"][0]

        plt.plot(hull_x, hull_y, 'b--', lw=2)
        plt.plot(x, grad_warm_edge * x + intercept_warm_edge, label = "Warm edge")
        plt.axhline(cold_edge)
        plt.axvline(full_veg_cover)

        temperatures_data = soil_canopy_temperatures(x,
                                                    y,
                                                    cold_edge,
                                                    grad_warm_edge,
                                                    intercept_warm_edge,
                                                    full_veg_cover
                                                    )

        common_data["T_SOIL"] = temperatures_data["T_soil_extreme"]
        common_data["T_CANOPY"] = temperatures_data["T_canopy_extreme"]

        # Grad and intercept for ALL points!
        common_data["gradient"] = temperatures_data["gradient_of_point"].values
        common_data["intercept"] = temperatures_data["intercept_of_point"].values

        common_data["p_o"], common_data["p_5"] = dummy_line(
            common_data["gradient"],common_data["intercept"])

        poly = Polygon((x, y) for x, y in zip(hull_x, hull_y))

        results = list(map(lambda p: interceptor(poly=poly, p_0 = p[0], p_5 = p[1], TSURF =p[2]),
                           zip(common_data["p_o"], common_data["p_5"], common_data["TSURF"])))

        common_data["T_soil_hull"], common_data["T_canopy_hull"] = zip(*results)

        merged_geo = retrieve_LPRM(common_data,
                      list_bbox,
                      target_res,
                      path_aux,
                      sat_sensor,
                      sat_band
                      )

        cbar_lut = {
            "TSURF": (270, 330),
            "T_soil_hull": (270, 330),
            "T_canopy_hull": (270, 330),
            # f"SM_{sat_band}": (0, 0.5),
            # f"SM_ADJ": (0, 0.5),
            # f"DIF_SM{sat_band}-ADJ": (-0.25, 0.25),
            # sat_band: (0, 0.5),
        }

        plot_maps_LPRM(merged_geo, cbar_lut, d)
        # plot_maps_day_night(merged_geo, night_LPRM, sat_band,)

        dt_original_array = merged_geo[f"SM_{sat_band}"].expand_dims(time = [d.date()])
        dt_original_ts.append(dt_original_array)

        dt_adjusted_array = merged_geo[f"SM_ADJ"].expand_dims(time = [d.date()])
        dt_adjusted_ts.append(dt_adjusted_array)

        nt_arr = night_LPRM[f"SM_{sat_band}"].expand_dims(time = [d.date()])
        nt_ts.append(nt_arr)

    except Exception as e:
        print(e)
        continue

dt_ori_ds = xr.concat(dt_original_ts, dim="time")
dt_adj_ds = xr.concat(dt_adjusted_ts, dim="time")
nt_ds = xr.concat(nt_ts, dim="time")

##
lat = 37.555028632
lon = -102.313477769

# plot_timeseries(dt_ori_ds, dt_adj_ds, nt_ds,lat,lon,sat_band = sat_band)


