import os

import contextily as ctx
import pandas as pd
import seaborn as sns
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject

plotlim_lut = {
    "R" : [-1,1],
    "BIAS" : [-0.35,0.35],
    "urmsd" : [0,0.2],
}
diff_lut = {
    "BIAS":[-0.15, 0.15]
}
metrics = ['BIAS', 'R', 'urmsd']


def plot_map(day, night, overpass, freq, diff=False):
  day_reset = day.reset_index()
  night_reset = night.reset_index()

  # Merge day and night dataframes on their shared location coordinates
  merged_df = pd.merge(
      day_reset, night_reset, on=['lat', 'lon'], suffixes=('_day', '_night')
  )
  metric = 'BIAS'

  # Calculate Day minus Night difference
  merged_df[f'{metric}_diff'] = (
      merged_df[f'{metric}_day'] - merged_df[f'{metric}_night']
  )

  # Clip latitudes slightly to prevent mercantile projection out-of-bounds errors (e.g., 90.00000000001)
  merged_df['lat'] = merged_df['lat'].clip(-85.0, 85.0)

  bbox = [-129.889527, -63.556488, 22.453091, 50.67319]
  fig = plt.figure(figsize=(14, 8))

  # Use Web Mercator projection so high-res tiles align correctly
  ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
  ax.set_extent(bbox, crs=ccrs.PlateCarree())

  plot_data = (
      merged_df[f'{metric}_{overpass}']
      if not diff
      else merged_df[f'{metric}_diff']
  )
  lut = plotlim_lut if not diff else diff_lut

  # Scatter plot points mapped using EPSG:4326 geographic coordinates
  sc = ax.scatter(
      merged_df['lon'],
      merged_df['lat'],
      c=plot_data,
      cmap='coolwarm',
      s=45,
      edgecolors='k',
      vmin=lut[metric][0],
      vmax=lut[metric][1],
      linewidth=0.4,
      transform=ccrs.PlateCarree(),
  )

  # Add high-res satellite tiles using the correct EPSG string for Web Mercator (EPSG:3857)
  ctx.add_basemap(
      ax,
      crs='EPSG:3857',
      source=ctx.providers.Esri.WorldImagery,
      zoom=6,
  )

  ax.gridlines(
      draw_labels=True,
      dms=True,
      x_inline=False,
      y_inline=False,
      linewidth=0.5,
      color='white',
      alpha=0.6,
  )

  cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.08, shrink=0.6)
  cbar.set_label(f'{metric}', fontsize=11)

  title_ = (
      f'{metric} {overpass} {freq}'
      if not diff
      else f'Day-Night {metric} {freq}'
  )
  plt.title(title_, fontsize=14, fontweight='bold', pad=15)
  plt.tight_layout()
  plt.show()


def plot_KG(day, night):

    # Ensure condition tags are set and datasets are combined
    night['Overpass'] = 'Night'
    day['Overpass'] = 'Day'
    combined_df = pd.concat([night, day], ignore_index=True)

    # Comprehensive mapping dictionary for Köppen-Geiger climate codes to full names
    climate_names = {
        'Af': 'Tropical Rainforest',
        'Am': 'Tropical Monsoon',
        'As': 'Tropical Savanna (Dry Summer)',
        'Aw': 'Tropical Savanna (Dry Winter)',
        'BWk': 'Arid Desert (Cold)',
        'BWh': 'Arid Desert (Hot)',
        'BSk': 'Arid Steppe (Cold)',
        'BSh': 'Arid Steppe (Hot)',
        'Csa': 'Temperate (Dry, Hot Summer)',
        'Csb': 'Temperate (Dry, Warm Summer)',
        'Csc': 'Temperate (Dry, Cold Summer)',
        'Cwa': 'Temperate (Dry Winter, Hot Summer)',
        'Cwb': 'Temperate (Dry Winter, Warm Summer)',
        'Cwc': 'Temperate (Dry Winter, Cold Summer)',
        'Cfa': 'Temperate (Fully Humid, Hot Summer)',
        'Cfb': 'Temperate (Fully Humid, Warm Summer)',
        'Cfc': 'Temperate (Fully Humid, Cold Summer)',
        'Dsa': 'Continental (Dry Summer, Hot Summer)',
        'Dsb': 'Continental (Dry Summer, Warm Summer)',
        'Dsc': 'Continental (Dry Summer, Cold Summer)',
        'Dsd': 'Continental (Dry Summer, Very Cold Winter)',
        'Dwa': 'Continental (Dry Winter, Hot Summer)',
        'Dwb': 'Continental (Dry Winter, Warm Summer)',
        'Dwc': 'Continental (Dry Winter, Cold Summer)',
        'Dwd': 'Continental (Dry Winter, Very Cold Winter)',
        'Dfa': 'Continental (Fully Humid, Hot Summer)',
        'Dfb': 'Continental (Fully Humid, Warm Summer)',
        'Dfc': 'Continental (Fully Humid, Cold Summer)',
        'Dfd': 'Continental (Fully Humid, Very Cold Winter)',
        'ET': 'Polar Tundra',
        'EF': 'Polar Eternal Frost',
    }

    # Map the short codes to full descriptions, keeping unmapped codes as-is
    combined_df['Climate_Full'] = combined_df['climate_KG'].map(climate_names)
    combined_df['Climate_Full'] = combined_df['Climate_Full'].fillna(
        combined_df['climate_KG']
    )

    # Define master hierarchical order
    master_climate_order = [
        'Tropical Rainforest',
        'Tropical Monsoon',
        'Tropical Savanna (Dry Summer)',
        'Tropical Savanna (Dry Winter)',
        'Arid Desert (Hot)',
        'Arid Desert (Cold)',
        'Arid Steppe (Hot)',
        'Arid Steppe (Cold)',
        'Temperate (Dry, Hot Summer)',
        'Temperate (Dry, Warm Summer)',
        'Temperate (Dry, Cold Summer)',
        'Temperate (Dry Winter, Hot Summer)',
        'Temperate (Dry Winter, Warm Summer)',
        'Temperate (Dry Winter, Cold Summer)',
        'Temperate (Fully Humid, Hot Summer)',
        'Temperate (Fully Humid, Warm Summer)',
        'Temperate (Fully Humid, Cold Summer)',
        'Continental (Dry Summer, Hot Summer)',
        'Continental (Dry Summer, Warm Summer)',
        'Continental (Dry Summer, Cold Summer)',
        'Continental (Dry Summer, Very Cold Winter)',
        'Continental (Dry Winter, Hot Summer)',
        'Continental (Dry Winter, Warm Summer)',
        'Continental (Dry Winter, Cold Summer)',
        'Continental (Dry Winter, Very Cold Winter)',
        'Continental (Fully Humid, Hot Summer)',
        'Continental (Fully Humid, Warm Summer)',
        'Continental (Fully Humid, Cold Summer)',
        'Continental (Fully Humid, Very Cold Winter)',
        'Polar Tundra',
        'Polar Eternal Frost',
    ]

    # Filter order to only include categories actually present in the dataset
    present_climates = combined_df['Climate_Full'].unique()
    climate_order = [c for c in master_climate_order if c in present_climates]
    # Add any unmapped or unexpected climate types cleanly at the end if present
    extra_climates = [
        c for c in present_climates if c not in master_climate_order
    ]
    climate_order.extend(extra_climates)


    for metric in metrics:
        plt.figure(figsize=(20, 10))
        sns.boxplot(
            data=combined_df,
            x='Climate_Full',
            y=metric,
            hue='Overpass',
            order=climate_order,
            palette={'Day': 'orange', 'Night': 'royalblue'},
        )
        plt.title(
            f'ISMN validation {metric} AMSR2 {freq} 2024',
            fontsize=13,
            fontweight='bold',
        )
        plt.xlabel('', fontsize=15)
        plt.ylim(plotlim_lut[metric][0],plotlim_lut[metric][1])
        plt.ylabel(metric, fontsize=15)
        plt.xticks(rotation=60, ha='right', fontsize=15)
        plt.legend(title='Overpass')
        plt.tight_layout()
        plt.show()


def plot_LC(day,night,freq):

    night['Overpass'] = 'Night'
    day['Overpass'] = 'Day'
    combined_df = pd.concat([night, day], ignore_index=True)

    # Comprehensive mapping dictionary for CCI Land Cover 2010 codes to full names/labels
    lc_2010_names = {
    0: 'No Data',
    10: 'Cropland, rainfed',
    11: 'Cropland, rainfed',
    12: 'Cropland, rainfed',
    20: 'Cropland, irrigated',
    30: 'Mosaic cropland (>50%) / vegetation (<50%)',
    40: 'Mosaic vegetation(>50%) / cropland (<50%)',
    50: 'Tree cover, broadleaved evergreen',
    60: 'Tree cover, broadleaved deciduous',
    61: 'Tree cover, broadleaved deciduous',
    70: 'Tree cover, needleleaved evergreen',
    80: 'Tree cover, needleleaved deciduous',
    90: 'Tree cover, mixed leaf type',
    100: 'Mosaic tree and shrub (>50%)',
    110: 'Mosaic herbaceous (>50%)',
    120: 'Shrubland',
    130: 'Grassland',
    140: 'Lichens and mosses',
    150: 'Sparse vegetation (<15%)',
    160: 'Tree cover, flooded, fresh or brackish water',
    170: 'Tree cover, flooded, saline water',
    180: 'Shrub or herbaceous coverr',
    190: 'Urban areas',
    200: 'Bare areas',
    210: 'Water bodies',
    220: 'Permanent snow and ice',
    }

    # Define the exact numerical order based on your value key
    lc_2010_order_keys = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    170,
    180,
    190,
    200,
    210,
    220,
    ]
    lc_2010_order = [lc_2010_names[k] for k in lc_2010_order_keys if k in lc_2010_names]

    # Map the numeric codes in lc_2010 to full descriptions
    combined_df['LC_2010_Full'] = combined_df['lc_2010'].map(lc_2010_names)
    combined_df['LC_2010_Full'] = combined_df['LC_2010_Full'].fillna(
    combined_df['lc_2010'].astype(str)
    )

    # Filter order to only include categories actually present in the dataset
    present_lc = combined_df['LC_2010_Full'].unique()
    lc_order = [c for c in lc_2010_order if c in present_lc]
    extra_lc = [c for c in present_lc if c not in lc_2010_order]
    lc_order.extend(extra_lc)


    for metric in metrics:
        plt.figure(figsize=(20, 12))
        sns.boxplot(
          data=combined_df,
          x='LC_2010_Full',
          y=metric,
          hue='Overpass',
          order=lc_order,
          palette={'Day': 'orange', 'Night': 'royalblue'},
        )
        plt.title(
            f'ISMN validation {metric} AMSR2 {freq} 2024',
            fontsize=13,
            fontweight='bold',
        )
        plt.xlabel('', fontsize=11)
        plt.ylabel(metric, fontsize=15)
        plt.ylim(plotlim_lut[metric][0],plotlim_lut[metric][1])
        plt.yticks( fontsize=15)
        plt.xticks(rotation=65, ha='right', fontsize = 15)
        plt.legend(fontsize=15,)
        plt.tight_layout()
        plt.show()


def get_dataframe(da, overpass):

    stacked_ds = da.stack(grid_point=("lat", "lon"))
    clean_stacked = stacked_ds.dropna(dim="grid_point", how="any")
    df = clean_stacked.to_dataframe()
    df['Overpass'] = overpass

    return df


def import_single_obj(
        filename_test,
        variable,
        root_path,
):

    filename = f"0-ISMN.soil_moisture_with_1-{filename_test}.{variable}.nc"

    path = os.path.join(root_path, filename,)
    dataset_ref = os.path.join(os.getcwd(), 'data', path)
    plot_obj = CustomPlotObject(dataset_ref)

    return plot_obj


def clean_data(obj):

    _df = obj.df
    df_cleaned = _df[~_df.index.duplicated(keep="first")]
    _xr = df_cleaned.to_xarray()

    return _xr


def rename_vars(da):

    target_list = [
        "R_between",
        "BIAS_between",
        "RMSD_between",
        "urmsd_between",
    ]
    da_new = da.rename_vars(
        {var : var.split("_")[0] for var in da.var() if any(target in var for target in target_list)}
    )

    return da_new


