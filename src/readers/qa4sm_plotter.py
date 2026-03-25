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
    "BIAS": (-0.1, 0.1),
    "R" : (-1,1),
    "urmsd": (0,0.35),
    "status":(None,None)
}


##

def import_single_obj(filename,
                      root_path= path_datasets):

    path = os.path.join(root_path, filename,)
    dataset_ref = os.path.join(os.getcwd(), 'data', path)
    plot_obj = CustomPlotObject(dataset_ref)

    return plot_obj



def obj_masker(obj_ref,
               obj_mask,
               var,
               ):
    _obj_ref = copy.copy(obj_ref)
    _obj_mask = copy.copy(obj_mask)

    ref_variables_string = [col for col in _obj_ref.df.columns if var in col][0]
    xr_ref = _obj_ref.df[ref_variables_string]

    mask_variable_string = [col for col in _obj_mask.df.columns if var in col][0] # the ugliest code ive ever done
    xr_mask = _obj_mask.df[mask_variable_string]

    x_ref_masked = xr_ref.mask(xr_mask.isna())

    _obj_ref.df[ref_variables_string] = x_ref_masked

    return _obj_ref


def qa_plotter(obj, ref_name, test_name, metric, value_range = None):

    obj.plot_map(metric = metric,
                      output_dir =output_path,
                      dataset_list = [ref_name,test_name],
                      title = f"{metric}:   {ref_name}  -  {test_name}",
                      value_range=value_range
                      )
    plt.show()


def histogram_plot(fname,
                   statistics,
                   xlim= [None,None],
                   maxval=None,
                   root_path = path_datasets,):

    data = xr.open_dataset(data_path)
    _stat_data = data[statistics]
    stat_data = _stat_data.values.ravel()

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

    plt.tight_layout()
    plt.show()

band_current = "c1"
era_var = "swvl1"
ref_type = "LPRM"
reference_dict = {"LPRM":f'SM{band_current}_NIGHT_ref',
                  "ERA5":f"ERA5_LAND.{era_var}"}

ref_name = reference_dict[ref_type]
test1_name = f'SM{band_current}_DAY_ref'
test2_name = f'SM{band_current}_DAY_regression'

fname_ref = f"0-{ref_name}.sm_with_1-{test1_name}.sm.nc"
fname_regression = f"0-{ref_name}.sm_with_1-{test2_name}.sm.nc"

metric=  "BIAS"

plot_obj_ref = import_single_obj(fname_ref)
plot_obj_regression = import_single_obj(fname_regression)


plot_obj_ref_masked = obj_masker(obj_ref=plot_obj_ref,
                                obj_mask=plot_obj_regression,
                                 var="BIAS")

qa_plotter(plot_obj_ref_masked,ref_name=ref_name,test_name=test1_name, metric=metric,
           value_range=[ plot_val_lut[metric][0], plot_val_lut[metric][1]])


qa_plotter(plot_obj_regression,ref_name=ref_name,test_name=test2_name, metric=metric,
           value_range=[ plot_val_lut[metric][0], plot_val_lut[metric][1]])


# histogram_plot(fname_ref,f"{metric}_between_0-{ref}_and_1-{test1}",
#                xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
#                # maxval  = 10000
#                )
# histogram_plot(fname_regression,f"{metric}_between_0-{ref}_and_1-{test2}",
#                xlim = [plot_val_lut[metric][0], plot_val_lut[metric][1]],
#                # maxval=10000
#                )

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


