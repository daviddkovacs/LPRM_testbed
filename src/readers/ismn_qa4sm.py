import copy
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import skew
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from LST.test_lprm_day import load_TB_daily, date_pattern_lut,file_pattern_lut
from LST.datacube_utilities import calc_Holmes_temp
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch


def import_single_obj(
        filename_test,
        variable,
        root_path,
):

    filename = f"0-ISMN.soil_moisture_with_1-{filename_test}.{variable}.nc"

    path = os.path.join(root_path, filename,)
    dataset_ref = os.path.join(os.getcwd(), 'data', path)
    plot_obj = CustomPlotObject(dataset_ref)

    return plot_obj


def clean_data(obj):

    _df = obj.df
    df_cleaned = _df[~_df.index.duplicated(keep="first")]
    _xr = df_cleaned.to_xarray()

    return _xr


def rename_vars(da):


    da_new = da.rename_vars({var : f"{var.split("_")[0]} {var.split("_")[1]}" for var in da.var()})



if __name__ == "__main__":

    freq = "C1"

    #DAY
    ob_day = import_single_obj(
        filename_test=f"AMSR2_day_2024_{freq}_float",
        variable=f"SM_{freq}",
        root_path="/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/validation/qa4sm"
    )

    #DAY
    ob_night = import_single_obj(
        filename_test=f"AMSR2_night_2024_{freq}_float",
        variable=f"SM_{freq}",
        root_path="/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/validation/qa4sm"
    )

    day = clean_data(ob_day)
    night = clean_data(ob_night)

    _day = rename_vars(day)

    night["BIAS_between_0-ISMN_and_1-AMSR2_day_2024_C1_float"].plot()
    plt.show()
    x =21