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


##

def import_plot_obj(filename, root_path = path_datasets ):

    dataset_name = os.path.join(root_path, filename,)

    dataset_path = os.path.join(os.getcwd(), 'data', dataset_name)

    plot_obj = CustomPlotObject(dataset_path)
    plot_obj.display_metrics_and_datasets()

    return plot_obj


def qa_plotter(fname,ref,test, metric, value_range = None):

    plot_obj = import_plot_obj(fname)
    plot_obj.plot_map(metric = metric,
                      output_dir =output_path,
                      dataset_list = [ref,test],
                      title = f"{metric}:   {ref}  -  {test}",
                      value_range=value_range
                      )
    plt.show()


def histogram_plot(fname,
                   statistics,
                   xlim= [None,None],
                   maxval=None,
                   root_path = path_datasets,):

    data_path = os.path.join(root_path,fname)
    data = xr.open_dataset(data_path)
    _stat_data = data[statistics]
    stat_data = _stat_data.values.ravel()

    _data_clean = stat_data[~np.isnan(stat_data)]
    data_clean = np.where((_data_clean>xlim[0]) & (_data_clean<xlim[1]), _data_clean,np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))

    n, bins, patches = ax.hist(data_clean, bins=250, color='#2c7bb6', edgecolor='white', alpha=0.9)

    mean_val = np.nanmean(data_clean)
    ax.axvline(mean_val, color='#d7191c', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.5f}')

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



ref = 'SMx_NIGHT_ref'
test1 = 'SMx_DAY_ref'
test2 = 'SMx_DAY_regression'

fname_ref = f"0-{ref}.sm_with_1-{test1}.sm.nc"
fname_regression = f"0-{ref}.sm_with_1-{test2}.sm.nc"


metric=  "RMSD"
qa_plotter(fname_ref, ref, test1, metric,
           value_range=(0.01,0.4)
           )
qa_plotter(fname_regression, ref, test2, metric,
           value_range=(0.01,0.4)
           )

histogram_plot(fname_ref,f"{metric}_between_0-{ref}_and_1-{test1}",
               xlim = [0,0.15],
               # maxval  = 10000
               )
histogram_plot(fname_regression,f"{metric}_between_0-{ref}_and_1-{test2}",
               xlim = [0,0.15],
               # maxval=10000

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

