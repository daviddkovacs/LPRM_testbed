from utilities.utils import mpdi, collocate_datasets
from utilities.plotting import scatter_plot, longitude_plot
from readers.Air import AirborneData
from readers.Sat import BTData
from utilities.plotting import scatter_density
from readers.ERA5 import ERA

import pandas as pd

def validator(ref_obj, test_obj, anc_obj ,
              comparison = None):

    # We load the dataframes
    air_pd = ref_obj.to_pandas()
    sat_pd = test_obj.to_pandas()

    # We get air & sat specific variables
    air_freq = ref_obj.air_freq
    sat_freq = test_obj.sat_freq

    bio_pd = anc_obj.to_pandas()
    bio_var = anc_obj.bio_var

    if comparison == "air2sat":
        ref_nn, test_nn = collocate_datasets(air_pd, sat_pd)
        test_nn["MPDI"] = mpdi(test_nn["bt_V"], test_nn["bt_H"])

        plot_var = "MPDI"
        xlabel = f"AMPR MPDI {air_freq} GHz"
        ylabel = f"AMSR2 MPDI {sat_freq}"

    if comparison == "air2bio":
        ref_nn, test_nn= collocate_datasets(air_pd, bio_pd)

        plot_var = "MPDI"
        xlabel = f"AMPR MPDI {air_freq} GHz"
        ylabel = f"AMSR2 MPDI {sat_freq}"

    if comparison == "sat2bio":

        ref_nn, test_nn= collocate_datasets(sat_pd, bio_pd)
        ref_nn["MPDI"] = mpdi(sat_pd["bt_V"], sat_pd["bt_H"])

        plot_var = bio_var
        xlabel = f"AMSR2 MPDI {sat_freq}"
        ylabel = f"ERA5 {bio_var}"



    scatter_plot(ref_nn[plot_var],
                 test_nn[plot_var],
                 xlabel=xlabel,
                 ylabel=ylabel,
                 xmax_val=0.1,
                 xmin_val=0,
                 ymax_val=0.1,
                 ymin_val=0)
    # scatter_density(ref_nn[plot_var],
    #              test_nn[plot_var],
    #              xlabel=xlabel,
    #              ylabel=ylabel)


    longitude_plot(ref_x= ref_nn["lon"] ,
                   ref_y = ref_nn["MPDI"],
                   test_x = test_nn["lon"],
                   test_y = test_nn["MPDI"],
                   # test2_x= bio_nn_air2era["lon"],
                   # test2_y = bio_nn_air2era[bio_var],
                   air_obj = ref_obj,
                   sat_obj = test_obj,
                   # bio_obj=anc_obj
                   )

def validator_all(path_air,
                  path_sat,
                  path_era,
                  sat_sensor = "AMSR2",
                  overpass = "night",
                  target_res = "10",
                  sat_freq = "10.7",
                  air_freq = "10.7",
                  bio_var = "lai_lv",
                  comparison = "air2sat"
                  ):

    datelist = ["2024-10-22", "2024-10-25", "2024-10-31"]
    flight_direction_list = ["WE", "EW"]
    scan_direction_list = ["1_25", "26_50"]

    ref_compound = pd.DataFrame({})
    test_compound = pd.DataFrame({})

    for d in datelist:
        for f in flight_direction_list:
            for s in scan_direction_list:

                air_pd = AirborneData(path=path_air,
                                          date=d,
                                          scan_direction=s,
                                          flight_direction=f,
                                          air_freq=air_freq,
                                          ).to_pandas()

                sat_pd = BTData(path=path_sat,
                                          sat_sensor=sat_sensor,
                                          date=d,
                                          overpass=overpass,
                                          target_res=target_res,
                                          sat_freq=sat_freq,
                                          ).to_pandas()

                bio_pd = ERA(path=path_era,
                             date=d,
                             bio_var=bio_var).to_pandas()

                if comparison == "air2sat":
                    # Airborne to Satellite
                    ref_nn, test_nn = collocate_datasets(air_pd, sat_pd)
                    ref_compound = pd.concat([ref_compound, ref_nn])
                    test_compound = pd.concat([test_compound, test_nn])
                    test_compound["MPDI"] = mpdi(test_compound["bt_V"], test_compound["bt_H"])

                    plot_var = "MPDI"
                    ylabel = f"AMSR2 MPDI {sat_freq}"

                if comparison == "air2bio":
                    # Airborne to ERA 5
                    ref_nn, test_nn = collocate_datasets(air_pd, bio_pd)
                    ref_compound = pd.concat([ref_compound, ref_nn])
                    test_compound = pd.concat([test_compound, test_nn])

                    plot_var = bio_var
                    ylabel = f"ERA5 {bio_var}"

                if comparison == "sat2bio":
                    # Satellite to ERA 5 variable
                    ref_nn, test_nn = collocate_datasets(sat_pd, bio_pd)
                    ref_compound  = pd.concat([ref_nn, ref_nn])
                    test_compound = pd.concat([test_nn, test_nn])

                    test_compound["MPDI"] = mpdi(test_compound["bt_V"], test_compound["bt_H"])

                    plot_var = bio_var
                    ylabel = f"ERA5 {bio_var}"


    scatter_plot(ref_compound["MPDI"],
                 test_compound[plot_var],
                 xlabel=f"AMPR MPDI {air_freq} GHz",
                 ylabel = ylabel,
                 )