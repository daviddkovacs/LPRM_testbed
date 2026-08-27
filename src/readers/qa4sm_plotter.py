import copy
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from LST.test_lprm_day import load_TB_daily, date_pattern_lut,file_pattern_lut
from LST.datacube_utilities import calc_Holmes_temp



path_datasets = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                 "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/qa4sm_netcdfs")

# dataset_name = os.path.join(path_datasets,"qa4sm_netcdfs", "
output_path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
               "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs")

hist_val_lut = {
    "BIAS": (-0.45, 0.45),
}

plot_val_lut = {
    "BIAS": (-0.25, 0.25),
    rf"$|\Delta$Biases|": (-0.1, 0.1),
    "R" : (-1,1),
    "urmsd": (0,0.20),
    "status":(None,None),
    "slope": (0.7,1.1),
    "intercept": (0,100),
}

color_lut = {
    "BIAS": "PiYG",
    rf"$|\Delta$Biases|": "coolwarm",
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

map_title_lut = {
    ""
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
                   variable = None,
                   title="",
                   freq = None
                   ):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 200,
    })
    values = plot_val_lut[metric]
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-170, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    if variable not in ["slope", "intercept"]:
        _variable = f'{metric}_between_0-{fname_ref}_and_1-{fname_test}'
    else:
        _variable = variable

    plot_da = dataset[_variable]
    if "BIAS" in _variable:
        plot_da = plot_da * (-1)
    if fname_test is not None:
        if "regression" in fname_test:
            plot_da = plot_da.where(values[0]+0.002 < plot_da)

    mesh = plot_da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x='lon',
        y='lat',
        cmap=color_lut[metric],
        vmin=values[0],
        vmax=values[1],
        add_colorbar=False,
        zorder=1
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
    cbar.set_label(f"{metric} {unit_lut[metric]}", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.title(title, fontsize=20, pad=15)

    plt.tight_layout(pad=0.5)
    plt.savefig(f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/"
                f"07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs/paper/maps/statistics_maps/{freq}/"
                f"{freq}_{metric}_{fname_ref}_{fname_test}", dpi=300, bbox_inches='tight')

    plt.show()

    return plot_da


def histogram_plot(obj,
                   obj2,
                   ref_name,
                   test_name,
                   metric,
                   label1="Data 1",
                   label2="Data 2",
                   xlim= [None,None],
                   maxval=None,
                   title = "",
                   xlabel = "",
                   freq = None
                   ):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 200,
    })

    statistics1 = f"{metric}_between_0-{ref_name}_and_1-{test_name[0]}"

    stat_data1 = obj.df[statistics1].values.ravel()
    data_clean1 = stat_data1[~np.isnan(stat_data1)]

    statistics2 = f"{metric}_between_0-{ref_name}_and_1-{test_name[1]}"
    stat_data2 = obj2.df[statistics2].values.ravel()
    data_clean2 = stat_data2[~np.isnan(stat_data2)]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot both histograms (alpha reduced to 0.6 so overlaps are visible)
    xlim = xlim if ref_type == "ERA5" else [-0.2,0.2]

    ax.hist(data_clean2, bins=150, range=(xlim[0], xlim[1]),
            color='#d7191c', edgecolor='white', alpha=0.99, label=label2)

    ax.hist(data_clean1, bins=150, range=(xlim[0], xlim[1]),
            color='#2c7bb6', edgecolor='white', alpha=0.6, label=label1)
    # Helper function to calculate stats
    def get_stats(data):
        return np.nanmean(data), np.nanvar(data), np.sqrt(np.nanvar(data)), len(data)

    m1, v1, s1, len1 = get_stats(data_clean1)
    m2, v2, s2, len2 = get_stats(data_clean2)

    # Format stats text for both datasets
    stats_text = (
        f'{label1} | Mean: {m1:.3g} | Std: {s1:.3g} | #: {len1}\n'
        f'{label2} | Mean: {m2:.3g} | Std: {s2:.3g} | #: {len2}'
    )

    print(title)
    print(stats_text)
    print("")

    # ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlabel(f"{metric} {unit_lut[metric]}", fontsize=12)

    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.legend(loc='upper right')  # explicitly added legend location so it doesn't overlap top-left text
    ax.set_xlim(xlim)
    ax.set_ylim([0, maxval])

    plt.savefig(f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                f"LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs/paper/histograms/{freq}/"
                f"{freq}_{metric}_{ref_name}_combined", dpi=300, bbox_inches='tight')

    plt.show()


def difference_maps(reference_xr,
               subtracted_xr,
               metric,
               fname_ref=None,
               fname_test=None,
               variable=None,
               title="",
               freq=None
               ):

    difference_xr = abs(reference_xr) - abs(subtracted_xr)
    values = plot_val_lut[metric]
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-170, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    if variable not in ["slope", "intercept"]:
        _variable = f'{metric}_between_0-{fname_ref}_and_1-{fname_test}'
    else:
        _variable = variable

    mesh = difference_xr.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x='lon',
        y='lat',
        cmap=color_lut[metric],
        vmin=values[0],
        vmax=values[1],
        add_colorbar=False,
        zorder=1
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
    cbar.set_label(f"{metric} {unit_lut[metric]}", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.title(title, fontsize=20, pad=15)

    plt.tight_layout(pad=0.5)
    # plt.savefig(f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/"
    #             f"07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs/paper/maps/statistics_maps/{freq}/"
    #             f"{freq}_{metric}_{ref_type}_difference", dpi=300, bbox_inches='tight')

    plt.show()


##

if __name__=="__main__":

    bands_to_plot = ["c1"]
    stats_to_plot = ["BIAS"]
    ref_type = "ERA5"

    for _band in bands_to_plot:
        for _metric in stats_to_plot:
            sm_var_name = {"LPRM": "sm",
                           "ERA5": "swvl1"}

            niceband_dict = {"x": "X",
                             "c1": "C"}

            ref_fname_dict = {"LPRM": f"SM{_band}_NIGHT_ref",
                              "ERA5": f"ERA5_LAND"}


            title_name_dict = {"LPRM": f"SM-{niceband_dict[_band]} Night Holmes",
                              "ERA5": f"ERA5 Land"}

            reference_filename = ref_fname_dict[ref_type]
            day_ref_filename = f"SM{_band}_DAY_ref"
            day_regression_filename = f"SM{_band}_DAY_regression"

            nice_day_ref_filename = f"SM Night Holmes ({niceband_dict[_band]}-band)"
            nice_day_regression_filename = f"SM Night Regression ({niceband_dict[_band]}-band)"
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
                           freq=_band
                           )

            plot_xr_test = manual_plotter(xr_test,
                           _metric,
                           fname_ref=reference_filename,
                           fname_test=day_regression_filename,
                           title=f"{nice_day_regression_filename} - {title_name_dict[ref_type]}",
                           freq=_band
                           )

            difference_maps(reference_xr=plot_xr_ref,
                            subtracted_xr=plot_xr_test,
                            metric=rf"$|\Delta$Biases|",
                            title=rf"{ref_type} reference",
                            freq=_band
                            )

            # histogram_plot(plot_obj_ref_masked,
            #                reference_filename,
            #                day_ref_filename,
            #                metric= _metric,
            #                xlim = [plot_val_lut[_metric][0], plot_val_lut[_metric][1]],
            #                maxval=12000,
            #                title= f"{_metric}: {reference_filename} v. {day_ref_filename}",
            #                freq = _band
            #
            #                )

            histogram_plot(plot_obj_ref_masked,
                           plot_obj_regression,
                           reference_filename,
                           [day_ref_filename,day_regression_filename],
                           metric= _metric,
                           xlim = [hist_val_lut[_metric][0], hist_val_lut[_metric][1]],
                           maxval=20000,
                           title= f"{ref_type} reference",
                           xlabel=f"{_metric}",
                           freq = _band,
                           label1= f"{_band} ref",
                           label2= f"{_band} regression"
                           )



## Maps of MPDI and their difference
  #   amsr2_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/01_resampled_bt/coarse_resolution/AMSR2/"
  #   zoomin_bbox =[
  #   -11.177304921271343,
  #   35.4538346353382,
  #   33.80649407930892,
  #   58.85815315416707
  # ]
  #   TB_DAY = xr.open_dataset(os.path.join(amsr2_path,"day/202405/amsr2_l1bt_day_20240501_25km.nc",),decode_timedelta=False).isel(time=0)
  #   TB_NIGHT = xr.open_dataset(os.path.join(amsr2_path,"night/202405/amsr2_l1bt_night_20240501_25km.nc"),decode_timedelta=False).isel(time=0)
  #
  #   MPDI_DAY = (TB_DAY["bt_6.9V"] - TB_DAY["bt_6.9H"]) / (TB_DAY["bt_6.9V"] + TB_DAY["bt_6.9H"])
  #   MPDI_NIGHT = (TB_NIGHT["bt_6.9V"] - TB_NIGHT["bt_6.9H"]) / (TB_NIGHT["bt_6.9V"] + TB_NIGHT["bt_6.9H"])
  #
  #   MPDI_DAY_ROI = crop2roi(MPDI_DAY, zoomin_bbox)
  #   MPDI_NIGHT_ROI = crop2roi(MPDI_NIGHT, zoomin_bbox)
  #   MPDI_dif = MPDI_DAY_ROI - MPDI_NIGHT_ROI
  #   MPDI_same =  xr.where(MPDI_dif <0.0001, True, False )
  #
  #
  #   fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5),
  #                            subplot_kw={'projection': ccrs.PlateCarree()})
  #
  #   vmin = 0
  #   vmax = 0.02
  #
  #   # --- Plot 1: Night ---
  #   ax1 = axes[0]
  #   ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
  #   ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
  #
  #   im = MPDI_NIGHT_ROI.plot.pcolormesh(
  #       ax=ax1,
  #       transform=ccrs.PlateCarree(),
  #       x='lon', y='lat',
  #       vmin=vmin, vmax=vmax,
  #       cmap='viridis',
  #       add_colorbar=False
  #   )
  #   ax1.set_title(f"MPDI Night")
  #
  #   gl1 = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
  #   gl1.top_labels = False
  #   gl1.right_labels = False
  #
  #   # --- Plot 2: Day ---
  #   ax2 = axes[1]
  #   ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
  #   ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
  #
  #   MPDI_DAY_ROI.plot.pcolormesh(
  #       ax=ax2,
  #       transform=ccrs.PlateCarree(),
  #       x='lon', y='lat',
  #       vmin=vmin, vmax=vmax,
  #       cmap='viridis',
  #       add_colorbar=False
  #   )
  #   ax2.set_title("MPDI Day")
  #
  #   gl2 = ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
  #   gl2.top_labels = False
  #   gl2.right_labels = False
  #   gl2.left_labels = False
  #
  #   # --- Plot 3: Same ---
  #   ax3 = axes[2]
  #   ax3.add_feature(cfeature.COASTLINE, linewidth=0.8)
  #   ax3.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
  #
  #   binary_cmap = ListedColormap(['white', 'darkgreen'])
  #
  #   im_same = MPDI_same.plot.pcolormesh(
  #       ax=ax3,
  #       transform=ccrs.PlateCarree(),
  #       x='lon', y='lat',
  #       vmin=0, vmax=1,  # Changed to span exactly 0 to 1
  #       cmap=binary_cmap,  # Use our new strict binary map
  #       add_colorbar=False
  #   )
  #   ax3.set_title("Difference between:\n"
  #                 "MPDI Night and MPDI Day")
  #
  #   gl3 = ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.0)
  #   gl3.top_labels = False
  #   gl3.right_labels = False
  #   gl3.left_labels = False
  #
  #   # ==========================================
  #   # 1. Adjust the main subplots to leave empty space at the bottom of the figure
  #   fig.subplots_adjust(bottom=0.25)
  #
  #   # 2. Add Independent Shared Colorbar for ax1 and ax2
  #   cbar_ax = fig.add_axes([0.29, 0.25, 0.18, 0.04])
  #   cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='MPDI', ticks=[0, 0.01, 0.02])
  #
  #   # 3. Add Legend for ax3 underneath the plot
  #   color_0 = im_same.cmap(im_same.norm(0))
  #   color_1 = im_same.cmap(im_same.norm(1))
  #
  #   patch_0 = mpatches.Patch(facecolor=color_0, edgecolor='black', label='Not equal')
  #   patch_1 = mpatches.Patch(facecolor=color_1, edgecolor='black', label='Equal')
  #
  #   # Changed: loc, bbox_to_anchor, and ncol
  #   ax3.legend(handles=[patch_1, patch_0],
  #              loc='upper center',  # Anchor point on the legend itself
  #              bbox_to_anchor=(0.5, -0.15),  # (x, y) coordinates relative to ax3
  #              ncol=2,  # Lay them out horizontally
  #              title='',
  #              framealpha=0.9)
  #   # plt.savefig("/home/ddkovacs/Desktop/mpdi_comparison.png", dpi=300, bbox_inches='tight')
  #
  #   plt.show()

