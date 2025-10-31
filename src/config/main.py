from utilities.utils import mpdi, collocate_datasets
from utilities.plotting import scatter_plot, longitude_plot
from readers.Air import AirborneData
from readers.Sat import SatelliteData
from readers.ERA5 import ERA

import pandas as pd

def validator(ref_obj, test_obj, anc_obj = None ):

    # We load the dataframes
    air_pd = ref_obj.to_pandas()
    sat_pd = test_obj.to_pandas()

    # We get air & sat specific variables
    air_freq = ref_obj.air_freq
    sat_freq = test_obj.sat_freq

    bio_obj = anc_obj
    bio_pd = bio_obj.to_pandas()
    bio_var = bio_obj.bio_var

    # Calc MPDI for satellite
    sat_pd["MPDI"] = mpdi(sat_pd["bt_V"], sat_pd["bt_H"])

    ref_nn_air2sat, test_nn_air2sat = collocate_datasets(air_pd, sat_pd)
    ref_nn_air2era, bio_nn_air2era = collocate_datasets(air_pd, bio_pd)

    scatter_plot(ref_nn_air2sat["MPDI"],
                 test_nn_air2sat["MPDI"],
                 xlabel=f"AMPR MPDI {air_freq} GHz",
                 ylabel=f"AMSR2 MPDI {sat_freq} GHz",)

    scatter_plot(ref_nn_air2era["MPDI"],
                 bio_nn_air2era[bio_var],
                 xlabel=f"AMPR MPDI {air_freq} GHz",
                 ylabel=f"ERA5 {bio_var}",
                 xmin_val=0,
                 xmax_val=0.2,
                 ymin_val=0,
                 ymax_val=5,
                 )

    longitude_plot(ref_x= air_pd["lon"] ,
                   ref_y = air_pd["MPDI"],
                   test_x = test_nn_air2sat["lon"],
                   test_y = test_nn_air2sat["MPDI"],
                   test2_x= bio_nn_air2era["lon"],
                   test2_y = bio_nn_air2era[bio_var],
                   air_obj = ref_obj,
                   sat_obj = test_obj,
                   bio_obj=anc_obj)

def validator_all(path_air,
                  path_sat,
                  path_era,
                  sat_sensor = "AMSR2",
                  overpass = "night",
                  target_res = "10",
                  sat_freq = "10.7",
                  air_freq = "10.7",
                  bio_var = "lai_lv",
                  ):



    datelist = ["2024-10-22", "2024-10-25", "2024-10-31"]
    flight_direction_list = ["WE", "EW"]
    scan_direction_list = ["1_25", "26_50"]

    air_mpdi_compound = pd.DataFrame({"lat": [], "lon": [], f"MPDI": []}) # air2sat
    sat_mpdi_compound = pd.DataFrame({"lat": [], "lon": [], f"MPDI": []}) # air2sat
    bio_compound = pd.DataFrame({"lat": [], "lon": [], bio_var: []}) # air2bio
    air_mpdi_compound_air2bio = pd.DataFrame({"lat": [], "lon": [],  f"MPDI": []}) # air2bio

    for d in datelist:
        for f in flight_direction_list:
            for s in scan_direction_list:

                air_pd = AirborneData(path=path_air,
                                          date=d,
                                          scan_direction=s,
                                          flight_direction=f,
                                          air_freq=air_freq,
                                          ).to_pandas()

                sat_pd = SatelliteData(path=path_sat,
                                          sat_sensor=sat_sensor,
                                          date=d,
                                          overpass=overpass,
                                          target_res=target_res,
                                          sat_freq=sat_freq,
                                          ).to_pandas()

                bio_pd = ERA(path=path_era,
                             date=d,
                             bio_var=bio_var).to_pandas()

                ref_nn_air2sat, test_nn_air2sat = collocate_datasets(air_pd, sat_pd)
                ref_nn_air2era, bio_nn_air2era = collocate_datasets(air_pd, bio_pd)

                air_mpdi_compound = pd.concat([air_mpdi_compound, ref_nn_air2sat])  # get filtered AMPR MPDI air2sat
                sat_mpdi_compound = pd.concat([sat_mpdi_compound, test_nn_air2sat])  # get filtered Satellite MPDI air2sat

                bio_compound = pd.concat([bio_compound,bio_nn_air2era])  # get filtered Satellite MPDI air2bio
                air_mpdi_compound_air2bio = pd.concat([air_mpdi_compound_air2bio, ref_nn_air2era])  # get filtered Satellite MPDI air2bio


    scatter_plot(air_mpdi_compound_air2bio["MPDI"],
                 bio_compound[bio_var],
                 xlabel=f"AMPR MPDI {air_freq} GHz",
                 ylabel=f"ERA5 {bio_var}",
                 xmin_val=0,
                 xmax_val=0.2,
                 ymin_val=0,
                 ymax_val=5,
                 )