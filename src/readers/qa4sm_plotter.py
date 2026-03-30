import copy

from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
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
    "BIAS": (-0.25, 0.25),
    "R" : (-1,1),
    "urmsd": (0,0.35),
    "status":(None,None)
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

    return _obj_ref


def qa_plotter(obj, ref_name, test_name, metric, value_range = None, title_additional = ""):
    obj.plot_map(metric = metric,
                      output_dir =None,
                      dataset_list = [ref_name,test_name],
                      title = f"{title_additional} {metric}:   {ref_name}  -  {test_name}",
                      value_range=value_range
                      )
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def histogram_plot(obj,
                   ref_name,
                   test_name,
                   metric,
                   xlim= [None,None],
                   maxval=None,
                   ):

    statistics = f"{metric}_between_0-{ref_name}_and_1-{test_name}"
    stat_data = obj.df[statistics].values.ravel()

    data_clean = stat_data[~np.isnan(stat_data)]
    # data_clean = np.where((data_nonan==0.0), np.nan, data_nonan)
    # data_clean = np.where((_data_clean>-0.001) & (_data_clean<0.001), np.nan,_data_clean,)

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

    # Place it at x=5%, y=95% of the plot area (top-left)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    ax.set_xlabel(statistics, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim([0,maxval])

    # plt.tight_layout()
    plt.show()


band_current = "c1"
ref_type = "ERA5"
metric=  "R"

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


plot_obj_ref_masked = obj_masker(obj_ref=plot_obj_ref,
                                obj_mask=plot_obj_regression,
                                 var=metric)

qa_plotter(plot_obj_ref_masked,
           ref_name=reference_filename,
           test_name=day_ref_filename,
           metric=metric,
           value_range=[plot_val_lut[metric][0], plot_val_lut[metric][1]],
           title_additional="2024")

qa_plotter(plot_obj_regression,
           ref_name=reference_filename,
           test_name=day_regression_filename,
           metric=metric,
           value_range=[plot_val_lut[metric][0], plot_val_lut[metric][1]],
           title_additional="2024")


histogram_plot(plot_obj_ref_masked,
               reference_filename,
               day_ref_filename,
               metric= metric,
               xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
               )

histogram_plot(plot_obj_regression,
               reference_filename,
               day_regression_filename,
               metric= metric,
               xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
               )

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


