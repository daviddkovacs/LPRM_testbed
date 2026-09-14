from insitu.ismn_utils import plot_map, plot_LC, get_dataframe, import_single_obj, clean_data, rename_vars


if __name__ == "__main__":

    # path = "/home/david/Desktop" Thinkpad
    path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/validation/qa4sm"
    freq = "C1"

    #DAY
    ob_day = import_single_obj(
        filename_test=f"AMSR2_day_2024_{freq}_float",
        variable=f"SM_{freq}",
        root_path=path
    )

    #DAY
    ob_night = import_single_obj(
        filename_test=f"AMSR2_night_2024_{freq}_float",
        variable=f"SM_{freq}",
        root_path=path
    )

    _day = clean_data(ob_day)
    _night = clean_data(ob_night)

    day = rename_vars(_day)
    night = rename_vars(_night)
    day_df = get_dataframe(day, "day")
    night_df = get_dataframe(night, "night")

##
    plot_map(day_df, night_df, overpass="day", freq=freq)
    plot_map(day_df, night_df, overpass="night", freq=freq)
    plot_map(day_df, night_df, overpass="night", freq=freq, diff=True)

    plot_LC(day_df, night_df, freq)