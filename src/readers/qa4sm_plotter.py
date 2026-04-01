import copy
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

path_datasets = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                 "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/qa4sm_netcdfs")

# dataset_name = os.path.join(path_datasets,"qa4sm_netcdfs", "
output_path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
               "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs")

plot_val_lut = {
    "BIAS": (-0.1, 0.1),
    "R" : (-1,1),
    "urmsd": (0,0.35),
    "status":(None,None),
    "slope": (0.7,1.1),
    "intercept": (0,100),
}

color_lut = {
    "BIAS": "PiYG",
    "R" : "RdBu_r",
    "urmsd": "PuOr",
    "status":(None,None),
    "slope":"RdYlGn",
    "intercept":"viridis",
}

##

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



def manual_plotter(dataset, metric, fname_ref=None, fname_test=None, variable = None, title=""):

    values = plot_val_lut[metric]
    fig = plt.figure(figsize=(8, 6))
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
    cbar.set_label(f"{metric}", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.title(title, fontsize=15, pad=15)
    plt.tight_layout()

    plt.show()


def histogram_plot(obj,
                   ref_name,
                   test_name,
                   metric,
                   xlim= [None,None],
                   maxval=None,
                   title = ""
                   ):

    statistics = f"{metric}_between_0-{ref_name}_and_1-{test_name}"
    stat_data = obj.df[statistics].values.ravel()

    data_clean = stat_data[~np.isnan(stat_data)]

    fig, ax = plt.subplots(figsize=(7, 5))

    n, bins, patches = ax.hist(data_clean, bins=250, range=(xlim[0], xlim[1]),color='#2c7bb6', edgecolor='white', alpha=0.9)

    mean_val = np.nanmean(data_clean)
    variance_val = np.nanvar(data_clean)
    std_val = np.sqrt(np.nanvar(data_clean))
    len_val = len(data_clean)
    stats_text = (
        f'Mean: {mean_val:.3}\n'
        f'Variance: {variance_val:.3}\n'
        f'Std: {std_val:.3}\n'
        f'#: {len_val}'
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    ax.set_xlabel(title, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim([0,maxval])

    # plt.tight_layout()
    plt.show()


if __name__=="__main__":

    band_current = "c1"
    ref_type = "LPRM"
    metric=  "urmsd"

    sm_var_name = {"LPRM" : "sm",
                   "ERA5" : "swvl1"}

    ref_fname_dict = {"LPRM": f"SM{band_current}_NIGHT_ref",
                      "ERA5": f"ERA5_LAND"}

    reference_filename = ref_fname_dict[ref_type]
    day_ref_filename = f"SM{band_current}_DAY_ref"
    day_regression_filename = f"SM{band_current}_DAY_regression"

    plot_obj_ref = import_single_obj(reference_filename,
                                     day_ref_filename,
                                     ref_type)

    plot_obj_regression = import_single_obj(reference_filename,
                                            day_regression_filename,
                                            ref_type)


    plot_obj_ref_masked, xr_ref, xr_test  = obj_masker(obj_ref=plot_obj_ref,
                                    obj_mask=plot_obj_regression,
                                     var=metric)


    manual_plotter(xr_ref,
                   metric,
                   fname_ref = reference_filename,
                   fname_test= day_ref_filename,
                   title=f"{metric}: {reference_filename} - {day_ref_filename}"
                   )

    manual_plotter(xr_test,
                   metric,
                   fname_ref=reference_filename,
                   fname_test=day_regression_filename,
                   title=f"{metric}: {reference_filename} - {day_regression_filename}"
                   )

    histogram_plot(plot_obj_ref_masked,
                   reference_filename,
                   day_ref_filename,
                   metric= metric,
                   xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
                   maxval=14000,
                   title= f"{metric}: {reference_filename} v. {day_ref_filename}",
                   # title= f"{metric}: LPRM Night v. {day_ref_filename}",
                   )

    histogram_plot(plot_obj_regression,
                   reference_filename,
                   day_regression_filename,
                   metric= metric,
                   xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
                   maxval=14000,
                   title= f"{metric}: {reference_filename} v. {day_regression_filename}"
                   )

##
    T_aux_path = ("/home/ddkovacs/shares/climers/Projects/"
                  "CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing/T_aux")

    regression_band = "c1"

    xr_taux = xr.open_dataset(os.path.join(T_aux_path,f"Daytime_T_aux_{regression_band}_MPDI0.01.nc"))
    regression_var = "intercept"
    manual_plotter(xr_taux,metric = regression_var,variable=regression_var,
                   title=f"{regression_band.upper()}-band MPDI trick \n  {regression_var} T$_{{simulated}}$-T$_{{KaV}}$")

##
metric_pretty_names = {
    'R': 'Pearson\'s r',
    'R_ci_lower': 'Pearson\'s r lower confidence interval',
    'R_ci_upper': 'Pearson\'s r upper confidence interval',
    'p_R': 'Pearson\'s r p-value',
    'RMSD': 'Root-mean-square deviation',
    'BIAS': 'Bias (difference of means)',
    'BIAS_ci_lower': 'Bias (difference of means) lower confidence interval',
    'BIAS_ci_upper': 'Bias (difference of means) upper confidence interval',
    'n_obs': '# observations',
    'urmsd': 'Unbiased root-mean-square deviation',
    'urmsd_ci_lower': 'Unbiased root-mean-square deviation lower confidence interval',
    'urmsd_ci_upper': 'Unbiased root-mean-square deviation upper confidence interval',
    'RSS': 'Residual sum of squares',
    'mse': 'Mean square error',
    'mse_corr': 'Mean square error correlation',
    'mse_bias': 'Mean square error bias',
    'mse_var': 'Mean square error variance',
    'snr': 'Signal-to-noise ratio',
    'snr_ci_lower': 'Signal-to-noise ratio lower confidence interval',
    'snr_ci_upper': 'Signal-to-noise ratio upper confidence interval',
    'err_std': 'Error standard deviation',
    'err_std_ci_lower': 'Error standard deviation lower confidence interval',
    'err_std_ci_upper': 'Error standard deviation upper confidence interval',
    'beta': 'TC scaling coefficient',
    'beta_ci_lower': 'TC scaling coefficient lower confidence interval',
    'beta_ci_upper': 'TC scaling coefficient upper confidence interval',
    'rho': 'Spearman\'s ρ',
    'rho_ci_lower': 'Spearman\'s ρ lower confidence interval',
    'rho_ci_uppper': 'Spearman\'s ρ upper confidence interval',
    'p_rho': 'Spearman\'s ρ p-value',
    'tau': 'Kendall rank correlation',
    'p_tau': 'Kendall tau p-value',
    'status': 'Validation success status',
    # 'tau': 'Kendall rank correlation',        # currently QA4SM is hardcoded not to calculate kendall tau
    # 'p_tau': 'Kendall tau p-value',
    'slopeR': 'Theil-Sen slope of R',
    'slopeURMSD': 'Theil-Sen slope of urmsd',
    'slopeBIAS': 'Theil-Sen slope of BIAS'
}


##
# check outliers:

ref = xr.open_dataset("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing/SM/MPDI_0.01/SMc1_DAY_ref.nc")
regression = xr.open_dataset("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing/SM/MPDI_0.01/SMc1_DAY_regression.nc")
slope = xr_taux["slope"]
intercept = xr_taux["intercept"]
ref_sm = ref["sm"].isel(time = slice(0,200))
regression_sm = regression["sm"].isel(time = slice(0,200))

##
bias = 10
lons = slice(1000-bias,1275+bias)
lats = slice(80-bias,91+bias)
focus_sm = regression_sm.isel(time = 2, lon= lons, lat= lats)
focus_slope = slope.isel(lon = lons, lat=  lats)
focus_intercept = intercept.isel(lon = lons, lat=  lats)


