from readers.Air import AirborneData
from readers.Sat import BTData
from readers.Bio import Bio, CLMS
from main import Plotter
import geopandas as gpd

if __name__ == "__main__":
    """
    #### Airborne Setup ####
    Frequencies:
        '10.7', '19.35', '37.1'
    Flight Directions:
        'EW', 'WE'
    Scan directions:
        '1_25', '26_50'


    #### Satellite Setup ####
    Frequencies (AMSR2):
        '6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0'
    Sensor:
        "amsr2" (more to come..)
    Target resolution:
        '10', '25' (kms)
    Overpass:
        'day', 'night'


    #### Common Setup ####
    date:
        '2024-10-22', '2024-10-25', '2024-10-31'

    """
    # Configure the parameters here ====================================================================================
    # Airborne (AMPR) variables
    path_air = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/WHYMSIE/data_from_RichDJ"
    air_freq =  '10.7'
    flight_direction = "EW"
    scan_direction = "26_50"

    # Satellite (AMSR2) variables
    path_sat = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/passive_input/medium_resolution/AMSR2"
    sat_freq = '18.7'
    sat_sensor = "amsr2"
    overpass = "day"
    target_res = "10"

    # ERA 5 variables (https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation)
    path_era = "/home/ddkovacs/shares/climers/Datapool/ECMWF_reanalysis/01_raw/ERA5-Land/datasets/images"
    bio_var = "stl1"

    #CLMS variables
    path_clms =  "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/WHYMSIE/ancillary_data/CLMS"
    clms_var = "LAI"

    # Comomn variables
    comparison = "air2sat" # "air2sat" or  "air2bio"
    date = "2024-10-22"

    single_validation = False
    compound_validation = True
    #  =================================================================================================================

    ER2_flight = AirborneData(path=path_air,
                              date=date,
                              scan_direction=scan_direction,
                              flight_direction=flight_direction,
                              air_freq=air_freq,
                              )

    AMSR2_OBS = BTData(path=path_sat,
                              sat_sensor=sat_sensor,
                              date=date,
                              overpass=overpass,
                              target_res=target_res,
                              sat_freq=sat_freq,
                              )
    ERA_SM  = Bio(path=path_era,
                  date =date,
                  bio_var=bio_var)

    CLMS_VEG = CLMS(path = path_clms,
                    bio_var=clms_var,
                    date=date)

    # era_pd = ERA_SM.to_pandas()
    # clms_pd = CLMS_VEG.to_pandas()


    p = Plotter(ER2_flight,
                AMSR2_OBS,
                CLMS_VEG,
                )

    # combined_data = p.get_data()

    if single_validation:
        p.scatterplot(comparison="air2bio")
        p.longitude_plot()

    if compound_validation:
        p.combined_scatter(comparison="air2bio")
