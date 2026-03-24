import matplotlib.pyplot as plt
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
import os
import matplotlib.pyplot as plt

path_datasets = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                 "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/qa4sm_netcdfs")


# dataset_name = os.path.join(path_datasets,"qa4sm_netcdfs", "
output_path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
               "LPRM/07_debug/daytime_retrieval/MPDI_trick/evaluation/figs")


##

def import_plot_obj(filename,root_path = path_datasets ):

    dataset_name = os.path.join(root_path, filename,)

    dataset_path = os.path.join(os.getcwd(), 'data', dataset_name)

    plot_obj = CustomPlotObject(dataset_path)
    plot_obj.display_metrics_and_datasets()

    return plot_obj


def qa_plotter(fname,ref,test, metric, value_range = None):
    plt.figure()
    plot_obj = import_plot_obj(fname)
    plot_obj.plot_map(metric = metric,
                      output_dir =output_path,
                      dataset_list = [ref,test],
                      title = f"{metric}:   {ref}  -  {test}",
                      value_range=value_range

                      )
    plt.show()

fname_regression = "0-SMx_NIGHT_ref.sm_with_1-SMx_DAY_regression.sm.nc"
fname_ref = "0-SMx_NIGHT_ref.sm_with_1-SMx_DAY_ref.sm.nc"


ref = 'SMx_NIGHT_ref'
test1 = 'SMx_DAY_ref'
test2 = 'SMx_DAY_regression'

qa_plotter(fname_ref, ref, test1, "BIAS",  value_range=(-0.15,0.15))
qa_plotter(fname_regression, ref, test2, "BIAS",  value_range=(-0.15,0.15))


