from utilities.utils import mpdi, collocate_datasets
from utilities.plotting import create_scatter_plot, create_longitude_plot
from readers.Air import AirborneData
from readers.Sat import BTData
from utilities.plotting import scatter_density
from readers.ERA5 import ERA

import pandas as pd

class Plotter:
    def __init__(self,ref_obj, test_obj, bio_obj =None,):

        # We load the dataframes
        self.ref_obj =ref_obj
        self.test_obj = test_obj

        self.ref_pd = ref_obj.to_pandas()
        self.test_pd = test_obj.to_pandas()

        # We get air & sat specific variables
        self.air_freq = ref_obj.air_freq
        self.flight_direction = ref_obj.flight_direction
        self.scan_direction = ref_obj.scan_direction

        self.sat_sensor = test_obj.sat_sensor
        self.sat_freq = test_obj.sat_freq
        self.target_res = test_obj.target_res
        self.target_res = test_obj.target_res

        if bio_obj:
            self.bio_obj = bio_obj
            self.bio_pd = bio_obj.to_pandas()
            self.bio_var = bio_obj.bio_var

    def scatterplot(self,
                    comparison = None,
                    ):

        if comparison == "air2sat":
            ref_nn, test_nn = collocate_datasets(self.ref_pd, self.test_pd)
            test_nn["MPDI"] = mpdi(test_nn["bt_V"], test_nn["bt_H"])

            ref_var = "MPDI"
            test_var = "MPDI"
            xlabel = f"AMPR MPDI {self.air_freq} GHz"
            ylabel = f"AMSR2 MPDI {self.sat_freq}"
            stat_text  = True

        if comparison == "air2bio":
            ref_nn, test_nn = collocate_datasets(self.ref_pd, self.bio_pd)

            ref_var = "MPDI"
            test_var = self.bio_var
            xlabel = f"AMPR MPDI {self.air_freq} GHz"
            ylabel = f"ERA5 {self.bio_var}"
            stat_text  = False

        create_scatter_plot(ref_nn[ref_var], test_nn[test_var], xlabel=xlabel, ylabel=ylabel, stat_text = stat_text)

    # scatter_density(ref_nn[plot_var],
    #              test_nn[plot_var],
    #              xlabel=xlabel,
    #              ylabel=ylabel)


    def longitude_plot(self):

        ref_nn, test_nn = collocate_datasets(self.ref_pd, self.test_pd)
        test_nn["MPDI"] = mpdi(test_nn["bt_V"], test_nn["bt_H"])

        ref_nn2bio, test_nnbio = collocate_datasets(self.ref_pd, self.bio_pd)

        kwargs_dict= {
            "sat_freq" : self.sat_freq,
            "sat_sensor" : self.sat_sensor,
            "target_res" : self.target_res,
            "flight_direction" : self.flight_direction,
            "air_freq" : self.air_freq,
            "bio_var" : self.bio_var
        }
        create_longitude_plot(self.ref_pd["lon"],
                              self.ref_pd["MPDI"],
                              test_nn["lon"],
                              test_nn["MPDI"],
                              test_nnbio["lon"],
                              test_nnbio[self.bio_var],
                              **kwargs_dict
                              )


    def combined_scatter(self,comparison = None):

        path_air = self.ref_obj.path
        path_sat = self.test_obj.path

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
                                              air_freq=self.ref_obj.air_freq,
                                              ).to_pandas()

                    sat_pd = BTData(path=path_sat,
                                              sat_sensor=self.test_obj.sat_sensor,
                                              date=d,
                                              overpass= self.test_obj.overpass,
                                              target_res=self.test_obj.target_res,
                                              sat_freq=self.test_obj.sat_freq,
                                              ).to_pandas()

                    bio_pd = ERA(path=self.bio_obj.path,
                                 date=d,
                                 bio_var=self.bio_obj.bio_var).to_pandas()

                    if comparison == "air2sat":
                        # Airborne to Satellite
                        ref_nn, test_nn = collocate_datasets(air_pd, sat_pd)
                        ref_compound = pd.concat([ref_compound, ref_nn])
                        test_compound = pd.concat([test_compound, test_nn])
                        test_compound["MPDI"] = mpdi(test_compound["bt_V"], test_compound["bt_H"])

                        ref_var = "MPDI"
                        test_var = "MPDI"
                        xlabel = f"AMPR MPDI {self.air_freq} GHz"
                        ylabel = f"AMSR2 MPDI {self.sat_freq}"
                        stat_text = True

                    if comparison == "air2bio":
                        # Airborne to ERA 5
                        ref_nn, test_nn = collocate_datasets(air_pd, bio_pd)
                        ref_compound = pd.concat([ref_compound, ref_nn])
                        test_compound = pd.concat([test_compound, test_nn])

                        ref_var = "MPDI"
                        test_var = self.bio_var
                        xlabel = f"AMPR MPDI {self.air_freq} GHz"
                        ylabel = f"ERA5 {self.bio_var}"
                        stat_text = False


        create_scatter_plot(ref_compound[ref_var], test_compound[test_var], xlabel=xlabel,
                            ylabel=ylabel, stat_text = stat_text)