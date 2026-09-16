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


path_datasets = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                 "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/qa4sm_netcdfs")

# dataset_name = os.path.join(path_datasets,"qa4sm_netcdfs", "
output_path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
               "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs")

hist_val_lut = {
    "LPRM": (-0.25, 0.25),
    "ERA5": (-0.45, 0.45),
}

plot_val_lut = {
    "BIAS": (-0.25, 0.25),
    rf"$|\Delta$Biases|": (0, 0.1),
    "R" : (-1,1),
    "urmsd": (0,0.20),
    "status":(None,None),
    "slope": (0.7,1.1),
    "intercept": (0,100),
}

color_lut = {
    "BIAS": "PiYG",
    rf"$|\Delta$Biases|": "BuGn",
    "R" : "RdBu_r",
    "urmsd": "YlGnBu",
    "status":(None,None),
    "slope":"RdYlGn",
    "intercept":"viridis",
}

unit_lut = {
    "BIAS": "[$m^3/m^3$]",
    rf"$|\Delta$Biases|": "[$m^3/m^3$]",
    "R" : "",
    "urmsd": "[$m^3/m^3$]",
}

maxval_lut = {
    "LPRM" : 15000,
    "ERA5": 20000
}



def import_single_obj(filename_ref,
                      filename_test,
                      ref_type,
                      root_path=path_datasets):

    sm_var_name = {"LPRM": "sm",
                   "ERA5": "swvl1"}

    filename = f"0-{filename_ref}.{sm_var_name[ref_type]}_with_1-{filename_test}.sm.nc"

    path = os.path.join(root_path, filename,)
    dataset_ref = os.path.join(os.getcwd(), 'data', path)
    plot_obj = CustomPlotObject(dataset_ref)

    return plot_obj


def obj_masker(obj_ref, obj_mask, var):
    _obj_ref = copy.copy(obj_ref)

    _obj_ref.df = _obj_ref.df.copy()

    ref_col = [col for col in _obj_ref.df.columns if col.startswith(f"{var}_between")][0]
    mask_col = [col for col in obj_mask.df.columns if col.startswith(f"{var}_between")][0]

    is_nan_mask = obj_mask.df[mask_col].isna()

    _obj_ref.df = _obj_ref.df.mask(is_nan_mask, axis=0)
    xr_ref =  _obj_ref.df.to_xarray()
    xr_test =  obj_mask.df.to_xarray()
    return _obj_ref, xr_ref,xr_test


def manual_plotter(dataset,
                   metric,
                   fname_ref=None,
                   fname_test=None,
                   variable=None,
                   title="",
                   ):


    values = plot_val_lut[metric]
    if ref_type == "LPRM":
        values = (-0.1,0.1)
    if variable not in ["slope", "intercept"]:
        _variable = f'{metric}_between_0-{fname_ref}_and_1-{fname_test}'
    else:
        _variable = variable

    plot_da = dataset[_variable]
    # if "BIAS" in _variable:
    #     plot_da = plot_da * (-1)
    if fname_test is not None:
        if "regression" in fname_test:
            # plot_da = plot_da.where(plot_da < 0.1)
            plot_da = plot_da

    # 2. Plotting Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })

    fig = plt.figure(figsize=(9, 5.5))

    # 3. GridSpec Layout
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 0.05],
        hspace=0.20,
        top=0.95,
        bottom=0.12,
        left=0.08,
        right=0.98
    )

    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    # Cartopy features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    plot_kwargs = {
        "cmap": color_lut[metric],
        "vmin": values[0],
        "vmax": values[1],
        "add_colorbar": False,
        "rasterized": True,
        "transform": ccrs.PlateCarree(),
        "zorder": 1
    }

    # 4. Draw the map
    mesh = plot_da.plot.pcolormesh(
        ax=ax,
        x='lon',
        y='lat',
        **plot_kwargs
    )

    # 5. Exactly as in double_world_plot
    ax.set_title(title, fontweight="bold", pad=8)

    # Force standard ticks so the axes aren't blank
    ax.set_xticks(np.arange(-180,180,30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-50,85,20), crs=ccrs.PlateCarree())

    # Labels and limits matching double_world_plot
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_ylim(-60, 85)
    ax.set_xlim(-170, 180)  # Retained your original x-limits

    # 6. Centered colorbar using a nested GridSpec
    cbar_gs = gs[1, 0].subgridspec(1, 3, width_ratios=[0.25, 0.5, 0.25])
    cax = fig.add_subplot(cbar_gs[0, 1])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend="both",ticks=np.linspace(values[0], values[1], 5)
    )
    cbar.set_label(f"{metric} {unit_lut[metric]}", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)


    # cbar.locator = MaxNLocator(nbins=5)  # Adjust nbins to show fewer ticks
    # cbar.update_ticks()
    plt.show()

    return plot_da




def histogram_plot(obj,
                   obj2,
                   ref_name,
                   test_name,
                   metric,
                   label1="Data 1",
                   label2="Data 2",
                   xlim=[None, None],
                   maxval=None,
                   title="",
                   xlabel="",
                   freq=None
                   ):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
    })

    statistics1 = f"{metric}_between_0-{ref_name}_and_1-{test_name[0]}"
    stat_data1 = obj.df[statistics1].values.ravel()
    data_clean1 = stat_data1[~np.isnan(stat_data1)]

    statistics2 = f"{metric}_between_0-{ref_name}_and_1-{test_name[1]}"
    stat_data2 = obj2.df[statistics2].values.ravel()
    data_clean2 = stat_data2[~np.isnan(stat_data2)]

    fig, ax = plt.subplots(figsize=(7, 5))

    xlim_range = xlim if (xlim[0] is not None and xlim[1] is not None) else [-0.2, 0.2]

    # Define consistent colors
    color1 = '#2c7bb6'
    color2 = '#d7191c'

    ax.hist(data_clean2, bins=150, range=(xlim_range[0], xlim_range[1]),
            color=color2, edgecolor='white', alpha=0.99)

    ax.hist(data_clean1, bins=150, range=(xlim_range[0], xlim_range[1]),
            color=color1, edgecolor='white', alpha=0.6)

    def get_stats(data):
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        median_val = np.nanmedian(data)
        skew_val = skew(data, nan_policy='omit')
        return mean_val, std_val, median_val, skew_val, len(data)

    m1, s1, med1, skew1, len1 = get_stats(data_clean1)
    m2, s2, med2, skew2, len2 = get_stats(data_clean2)

    if ref_type == "LPRM":
        stats_text1 = f'Mean: {m1:.3g} | Std: {s1:.3g}\nMedian: {med1:.3g} | Skew: {skew1:.3g}'
        stats_text2 = f'Mean: {m2:.3g} | Std: {s2:.3g}\nMedian: {med2:.3g} | Skew: {skew2:.3g}'
    if ref_type == "ERA5":
        stats_text1 = f'  Mean: {m1:.3g}\n  Std: {s1:.3g}\n  Median: {med1:.3g}\n  Skew: {skew1:.3g}'
        stats_text2 = f'  Mean: {m2:.3g}\n  Std: {s2:.3g}\n  Median: {med2:.3g}\n  Skew: {skew2:.3g}'

    print(title)
    print(f"{label1} | {stats_text1}")
    print(f"{label2} | {stats_text2}")
    print("")

    # Integrate statistics and color swatches directly into the legend
    legend_elements = [
        Patch(facecolor=color1, edgecolor='white', alpha=0.6, label=f"{label1}\n{stats_text1}"),
        Patch(facecolor=color2, edgecolor='white', alpha=0.99, label=f"{label2}\n{stats_text2}")
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, handlelength=1.5, handleheight=1.5)

    ax.set_xlabel(f"{metric}", fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.set_xlim(xlim_range)
    if maxval is not None:
        ax.set_ylim([0, maxval])

    plt.show()


def difference_maps(reference_xr,
                    test_xr,
                    metric,
                    fname_ref=None,
                    fname_test=None,
                    variable=None,
                    title="",
                    freq=None):

    difference_xr = abs(abs(test_xr)- abs(reference_xr))
    values = plot_val_lut[metric]

    if variable not in ["slope", "intercept"]:
        _variable = f'{metric}_between_0-{fname_ref}_and_1-{fname_test}'
    else:
        _variable = variable

    # 2. Plotting Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })

    fig = plt.figure(figsize=(9, 5.5))

    # 3. GridSpec Layout
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 0.05],
        hspace=0.20,
        top=0.95,
        bottom=0.12,
        left=0.08,
        right=0.98
    )

    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    # Cartopy features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    plot_kwargs = {
        "cmap": color_lut[metric],
        "vmin": values[0],
        "vmax": values[1],
        "add_colorbar": False,
        "rasterized": True,
        "transform": ccrs.PlateCarree(),
        "zorder": 1
    }

    # 4. Draw the map
    mesh = difference_xr.plot.pcolormesh(
        ax=ax,
        x='lon',
        y='lat',
        **plot_kwargs
    )

    # 5. Axes limits, ticks, and labels matching double_world_plot
    ax.set_title(title, fontweight="bold", pad=8)

    # Force standard ticks so the axes aren't blank
    ax.set_xticks(np.arange(-180,180,30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-50,85,20), crs=ccrs.PlateCarree())

    # Labels and limits
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_ylim(-60, 85)
    ax.set_xlim(-170, 180)

    # 6. Centered colorbar using a nested GridSpec
    cbar_gs = gs[1, 0].subgridspec(1, 3, width_ratios=[0.25, 0.5, 0.25])
    cax = fig.add_subplot(cbar_gs[0, 1])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend="both",
    )
    cbar.set_label(f"{metric} {unit_lut[metric]}", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)


    plt.show()


##

if __name__=="__main__":

    bands_to_plot = ["x"]
    stats_to_plot = ["BIAS"]
    ref_type = "LPRM"

    for _band in bands_to_plot:
        for _metric in stats_to_plot:
            sm_var_name = {"LPRM": "sm",
                           "ERA5": "swvl1"}

            niceband_dict = {"x": "X",
                             "c1": "C"}

            ref_fname_dict = {"LPRM": f"SM{_band}_NIGHT_ref",
                              "ERA5": f"ERA5_LAND"}


            title_name_dict = {"LPRM": f"SM Night Holmes ({niceband_dict[_band]}-band)",
                              "ERA5": f"ERA5 Land ({niceband_dict[_band]}-band)"}

            reference_filename = ref_fname_dict[ref_type]
            day_ref_filename = f"SM{_band}_DAY_ref"
            day_regression_filename = f"SM{_band}_DAY_regression"

            nice_day_ref_filename = f"SM Day Holmes"
            nice_day_regression_filename = f"SM Day Regression"
            plot_obj_ref = import_single_obj(reference_filename,
                                             day_ref_filename,
                                             ref_type)

            plot_obj_regression = import_single_obj(reference_filename,
                                                    day_regression_filename,
                                                    ref_type)


            plot_obj_ref_masked, xr_ref, xr_test  = obj_masker(obj_ref=plot_obj_ref,
                                            obj_mask=plot_obj_regression,
                                             var=_metric)


            plot_xr_ref = manual_plotter(xr_ref,
                           _metric,
                           fname_ref = reference_filename,
                           fname_test= day_ref_filename,
                           title=f"{nice_day_ref_filename} - {title_name_dict[ref_type]}",
                           )

            plot_xr_test = manual_plotter(xr_test,
                           _metric,
                           fname_ref=reference_filename,
                           fname_test=day_regression_filename,
                           title=f"{nice_day_regression_filename} - {title_name_dict[ref_type]}",
                           )

            difference_maps(reference_xr=plot_xr_test,
                            test_xr=plot_xr_ref,
                            metric=rf"$|\Delta$Biases|",
                            title=rf"{ref_type} reference ({niceband_dict[_band]}-band)",
                            freq=_band
                            )


            histogram_plot(plot_obj_ref_masked,
                           plot_obj_regression,
                           reference_filename,
                           [day_ref_filename,day_regression_filename],
                           metric= _metric,
                           xlim = [hist_val_lut[ref_type][0], hist_val_lut[ref_type][1]],
                           maxval=maxval_lut[ref_type],
                           title= f"{ref_type} reference ({niceband_dict[_band]}-band)",
                           xlabel=f"{_metric}",
                           freq = _band,
                           label1= f"Holmes",
                           label2= f"Regression"
                           )

