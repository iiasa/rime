# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:20:24 2023

@author: werning
"""

import sys

sys.path.append("H:\\git\\climate_impacts_processing")

import seaborn as sns
import xarray as xr
import numpy as np
import hotspots_functions_tables as hft
import hotspots_functions_pp as hfp
import os
import matplotlib.pyplot as plt
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import itertools as it
from scipy.interpolate import interp1d

from rime.process_config import *
from rime.rime_functions import *


input_dir = "C:\\Users\\{user}\\IIASA\\ECE.prog - Research Theme - NEXUS\\Hotspots_Explorer_2p0\\Data\\1_mmm_stats"
score_dir = "C:\\Users\\{user}\\IIASA\ECE.prog - Research Theme - NEXUS\\Hotspots_Explorer_2p0\\Data\\3_scores"
diff_dir = "C:\\Users\\{user}\\IIASA\ECE.prog - Research Theme - NEXUS\\Hotspots_Explorer_2p0\\Data\\2_differences"
std_dir = "C:\\Users\\{user}\\IIASA\\ECE.prog - Research Theme - NEXUS\\Hotspots_Explorer_2p0\\Data\\score_revision\\hist_standard_deviations_V2\\"
# plot_dir = "C:\\Users\\{user}\\IIASA\ECE.prog - Research Theme - NEXUS\\Hotspots_Explorer_2p0\\Data\\score_revision\\bivariate"
# yaml_path = "H:\\git\\climate_impacts_processing\\hotspots.yml"
# landmask_path = "H:\\git\\climate_impacts_processing\\landareamaskmap0.nc"
# kg_class_path = "H:\\git\\climate_impacts_processing\\kg_class.nc"

# Set interpolation & load landarea mask and parameters
land_mask = hfp.load_landmask(landmask_path)
params = hfp.load_parameters(yaml_path)
with xr.open_dataset(kg_class_path, engine="netcdf4") as kg_class:
    kg_class.load()
kg_class = kg_class.sel(band=0)

indicators = [
    "cdd",
    "dri",
    "dri_qtot",
    "ia_var",
    "ia_var_qtot",
    "precip",
    "sdd",
    "sdd_24p0",
    "sdd_20p0",
    "sdd_18p3",
    "seas",
    "seas_qtot",
    "tr20",
    "wsi",
]
# indicators = ['seas']
gmt_threshold = 3.0

# %% Functions
# -----------------------------------------------------------------------------


def bin_data(data, bins):
    data = xr.where(data > bins[3], bins[3], data)
    data = xr.where(data < bins[0], bins[0], data)

    m = interp1d(bins, [0, 1, 2, 3])
    pt = m(data.values)
    data.values = pt

    return data


# combination = [f'{ind}_{i}' for i in (variables)]

# indic_bins = {i: params[i]['score_range'] for i in variables}
# # indic_bins = {i: score_ranges[i] for i in combination}

# for var in variables:

#     # data[var] = xr.where(data[var] > max(indic_bins[f'{ind}_{var}']),
#     #                      max(indic_bins[f'{ind}_{var}']), data[var])
#     # data[var] = xr.where(data[var] < min(indic_bins[f'{ind}_{var}']),
#     #                      min(indic_bins[f'{ind}_{var}']), data[var])

#     # m = interp1d(indic_bins[f'{ind}_{var}'], score_bins)

#     data[var] = xr.where(data[var] > max(indic_bins[var]),
#                          max(indic_bins[var]), data[var])
#     data[var] = xr.where(data[var] < min(indic_bins[var]),
#                          min(indic_bins[var]), data[var])

#     m = interp1d(indic_bins[var], score_bins)


#     pt = m(data[var].values)
#     data[var].values = pt

# return data

# %% Calculation
# -----------------------------------------------------------------------------

plot_list = []

for ind in indicators:
    # Select mean, apply land mask and remove inf
    historical = xr.open_dataset(
        os.path.join(
            input_dir,
            "Historical",
            f'ISIMIP{params["protocol"][ind]}_MM_historical_{ind}.nc4',
        )
    )
    historical = hfp.apply_land_mask(historical, land_mask)
    historical = historical.where(historical.pipe(np.isfinite)).fillna(np.nan)

    # Select mean, apply land mask and remove inf
    future = xr.open_dataset(
        os.path.join(input_dir, f'ISIMIP{params["protocol"][ind]}_MM_{ind}_0p1.nc4')
    )
    future = hfp.apply_land_mask(future, land_mask)
    future = future.where(future.pipe(np.isfinite)).fillna(np.nan)

    if ind == "seas" or ind == "seas_qtot":
        historical = historical.where((historical < 5) & (historical > -5))
        future = future.where((future < 5) & (future > -5))

    if ind == "ia_var" or ind == "ia_var_qtot":
        historical = historical.where((historical < 2) & (historical > -2))
        future = future.where((future < 2) & (future > -2))

    # Load score data for plotting
    scores = xr.open_dataset(
        os.path.join(score_dir, f'ISIMIP{params["protocol"][ind]}_Scores_{ind}_0p1.nc4')
    )

    # Load diff data for plotting
    diff = xr.open_dataset(
        os.path.join(diff_dir, f'ISIMIP{params["protocol"][ind]}_Diff_{ind}_0p1.nc4')
    )
    diff = diff.where(diff.pipe(np.isfinite)).fillna(np.nan)

    for var in params["indicators"][ind]:
        # Select variable and rename it
        hist_data = historical[var][1].rename("hist")
        plot_hist = hist_data.hvplot(
            x="lon",
            y="lat",
            shared_axes=False,
            cmap=params["indicators"][ind][var]["ind_cmap"],
            clim=(
                params["indicators"][ind][var]["ind_min"],
                params["indicators"][ind][var]["ind_max"],
            ),
            title="Absolute - historical",
        )

        # Select variable and rename it
        future_data = future.sel({"threshold": gmt_threshold})
        future_data = future_data[var][1, :, :].rename(
            str(gmt_threshold).replace(".", "p")
        )
        # plot_future = future_data.hvplot(x='lon', y='lat', shared_axes=False, cmap = params["indicators"][ind][var]['ind_cmap'], clim=(params["indicators"][ind][var]['ind_min'], params["indicators"][ind][var]['ind_max']), title=f'Absolute - {str(gmt_threshold).replace(".", "p")}')

        # Select variable and rename it
        diff_data = diff.sel({"threshold": gmt_threshold})
        diff_data = diff_data[var][0].rename("diff")
        # plot_diff = diff_data.hvplot(x='lon', y='lat', shared_axes=False, cmap=params["indicators"][ind][var]['diff_cmap'], clim=(params["indicators"][ind][var]['diff_min'], params["indicators"][ind][var]['diff_max']), title=f'Difference - {str(gmt_threshold).replace(".", "p")}')

        # Select variable and rename it
        score_data = scores.sel({"threshold": gmt_threshold})
        score_data = score_data[var][0].rename("score")
        # plot_score = score_data.hvplot(x='lon', y='lat', cmap='magma_r', shared_axes=False, title=f'Score - {str(gmt_threshold).replace(".", "p")}')

        # Load standard deviation data, apply land mask, remove inf and rename
        std_dev = hft.load_netcdf(
            os.path.join(
                std_dir,
                f'{params["indicators"][ind][var]["short_name"]}_hist_std_GCM_avg.nc',
            )
        )
        std_dev = hfp.apply_land_mask(std_dev, land_mask)
        std_dev = std_dev.where(std_dev.pipe(np.isfinite)).fillna(np.nan)
        std_dev = std_dev.rename("std")

        # Calculate change in standard deviation
        std_change = (future_data - hist_data) / std_dev
        std_change = std_change.where(std_change.pipe(np.isfinite)).fillna(np.nan)
        std_change = std_change.rename("std_change")
        # plot_z = std_change.hvplot(x='lon', y='lat', cmap='magma_r', clim=(0,3), shared_axes=False, title='Z score')

        # Joint plots
        # hist_plot = sns.jointplot(x=hist_data.to_dataframe()['hist'], y=std_change.to_dataframe()['std_change'])
        # hist_plot.ax_marg_x.set_xlim(0, 1800)
        # hist_plot.ax_marg_y.set_ylim(0, 50)
        # hist_plot.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_joint_hist_std_change.png'))
        # future_plot = sns.jointplot(x=future_data.to_dataframe()[str(gmt_threshold).replace(".", "p")], y=std_change.to_dataframe()['std_change'])
        # future_plot.ax_marg_x.set_xlim(0, 1800)
        # future_plot.ax_marg_y.set_ylim(0, 50)
        # future_plot.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_joint_{str(gmt_threshold).replace(".", "p")}_std_change.png'))

        joint_merged = xr.merge([hist_data, std_change]).to_dataframe()
        # plot_joint_hist = joint_merged.hvplot.scatter(x='hist', y='std_change', shared_axes=False, cmap=['blue'], title='Joint plot hist - z score')

        # Bin data
        std_change_binned = bin_data(std_change, [0, 1, 2, 3])

        # Calculate quartiles
        bins = hist_data.quantile([0, 0.25, 0.5, 0.75])

        # Quartile bins
        hist_data_binned = bin_data(hist_data, bins)
        future_data_binned = bin_data(future_data, bins)

        # Create bivariate score
        hist_bivariate = std_change_binned + hist_data_binned
        hist_bivariate = hist_bivariate.rename("std_score")
        future_bivariate = std_change_binned + future_data_binned
        future_bivariate = future_bivariate.rename("std_score")
        # plot_score_quartile = future_bivariate.hvplot(x='lon', y='lat', cmap='magma_r', shared_axes=False, clim=(0,6), title='Score - Quartiles')

        color_cycle = hv.Cycle(
            [
                "#fcfdbf",
                "#feb078",
                "#f1605d",
                "#b5367a",
                "#721f81",
                "#2c115f",
                "#000004",
            ]
        )
        explicit_mapping = {
            "0": "#fcfdbf",
            "1": "#feb078",
            "2": "#f1605d",
            "3": "#b5367a",
            "4": "#721f81",
            "5": "#2c115f",
            "6": "#000004",
        }

        #
        joint_hist_quartiles_merged = xr.merge(
            [hist_data, std_change, hist_bivariate]
        ).to_dataframe()
        # plot_joint_hist_quartiles = joint_hist_quartiles_merged.hvplot.scatter(x='hist', y='std_change', by='std_score', color=explicit_mapping, title='Joint plot hist - z score quartiles')
        joint_future_quartiles_merged = xr.merge(
            [future_data, std_change, future_bivariate]
        ).to_dataframe()
        # plot_joint_hist_quartiles = joint_hist_quartiles_merged.hvplot.scatter(x='hist', y='std_change', color='std_score', cmap=explicit_mapping, title='Joint plot hist - z score quartiles')
        # plot_joint_future_quartiles = joint_future_quartiles_merged.hvplot.scatter(x=f'{str(gmt_threshold).replace(".", "p")}', y='std_change', color='std_score', cmap=explicit_mapping, title=f'Joint plot {str(gmt_threshold).replace(".", "p")} - z score quartiles')

        # plot_joint_hist_quartiles = joint_hist_quartiles_merged.hvplot.scatter(x='hist', y='std_change', color='std_score', cmap='magma_r', title='Joint plot hist - z score quartiles').redim.range(std_score=(0, 6))
        # plot_joint_future_quartiles = joint_future_quartiles_merged.hvplot.points(x=f'{str(gmt_threshold).replace(".", "p")}', y='std_change', color='std_score', cmap='magma_r', title=f'Joint plot {str(gmt_threshold).replace(".", "p")} - z score quartiles').redim.range(std_score=(0, 6))

        # # Joint plot - coloured with score
        # joint_hist = xr.merge([hist_data, std_change, hist_bivariate]).to_dataframe()
        # sns.jointplot(data=joint_hist, x='hist', y='std_change', c=joint_hist.std_score.dropna(), joint_kws={"color":None, 'cmap':'magma_r'})
        # joint_future = xr.merge([future_data, std_change, future_bivariate]).to_dataframe()
        # sns.jointplot(data=joint_future, x='2p0', y='std_change', c=joint_future.std_score.dropna(), joint_kws={"color":None, 'cmap':'magma_r'})

        kg_hist_all = xr.DataArray(
            data=np.full([len(hist_data.lat), len(hist_data.lon)], np.nan),
            coords={"lat": hist_data.lat, "lon": hist_data.lon},
        )
        kg_future_all = xr.DataArray(
            data=np.full([len(hist_data.lat), len(hist_data.lon)], np.nan),
            coords={"lat": hist_data.lat, "lon": hist_data.lon},
        )

        for k in range(1, 6):
            kg_hist_data = hist_data.where(kg_class == k).kg_class
            kg_future_data = future_data.where(kg_class == k).kg_class
            kg_std_change_binned = std_change_binned.where(kg_class == k).kg_class

            if var == "wsi":
                kg_bins = [0.1, 0.2, 0.3, 0.4]
            else:
                kg_bins = kg_hist_data.quantile([0, 0.25, 0.5, 0.75]).values
            print(f"{k} - {kg_bins}")
            kg_hist_data_binned = bin_data(kg_hist_data, kg_bins)
            kg_future_data_binned = bin_data(kg_future_data, kg_bins)

            kg_hist_bivariate = kg_std_change_binned + kg_hist_data_binned
            kg_future_bivariate = kg_std_change_binned + kg_future_data_binned

            kg_hist_all = xr.where(kg_class == k, kg_hist_bivariate, kg_hist_all)
            kg_future_all = xr.where(kg_class == k, kg_future_bivariate, kg_future_all)

            # fig = plt.figure()
            # kg_hist_bivariate.plot(cmap='magma_r', vmin=0, vmax=6)
            # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_kg_class_{k}_score_hist.png'))
            # plt.close()

            # fig = plt.figure()
            # kg_future_bivariate.plot(cmap='magma_r', vmin=0, vmax=6)
            # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_kg_class_{k}_score_{str(gmt_threshold).replace(".", "p")}.png'))
            # plt.close()

        # plot_kg_score_quartile = kg_future_all.hvplot(x='lon', y='lat', cmap='magma_r', shared_axes=False, clim=(0,6), title='Score - Quartiles - KG')
        plot_kg_score_quartile = kg_future_all.hvplot(
            x="lon",
            y="lat",
            cmap="magma_r",
            shared_axes=False,
            clim=(0, 6),
            title=f'{params["indicators"][ind][var]["long_name"]}',
        )

        # Create kg bivariate score
        kg_hist_all = kg_hist_all.kg_class.rename("std_score")
        kg_future_all = kg_future_all.kg_class.rename("std_score")

        #
        joint_hist_kg_quartiles_merged = xr.merge(
            [hist_data, std_change, kg_hist_all]
        ).to_dataframe()
        joint_future_kg_quartiles_merged = xr.merge(
            [future_data, std_change, kg_future_all]
        ).to_dataframe()
        # plot_joint_hist_kg_quartiles = joint_hist_kg_quartiles_merged.hvplot.scatter(x='hist', y='std_change', by='std_score', color=color_cycle, title='Joint plot hist - z score quartiles KG')
        # plot_joint_future_kg_quartiles = joint_future_kg_quartiles_merged.hvplot.scatter(x=f'{str(gmt_threshold).replace(".", "p")}', y='std_change', by='std_score', color=color_cycle, title=f'Joint plot {str(gmt_threshold).replace(".", "p")} - z score quartiles KG')
        # plot_joint_hist_kg_quartiles = joint_hist_kg_quartiles_merged.hvplot.scatter(x='hist', y='std_change', color='std_score', cmap='magma_r', title='Joint plot hist - z score quartiles KG').redim.range(std_score=(0, 6))
        # plot_joint_future_kg_quartiles = joint_future_kg_quartiles_merged.hvplot.scatter(x=f'{str(gmt_threshold).replace(".", "p")}', y='std_change', color='std_score', cmap='magma_r', title=f'Joint plot {str(gmt_threshold).replace(".", "p")} - z score quartiles KG').redim.range(std_score=(0, 6))

        # print('saving plot')
        # plot_list = [plot_future, plot_diff, plot_score,
        #               plot_joint_hist, plot_hist, plot_z,
        #               plot_joint_hist_quartiles, plot_joint_future_quartiles, plot_score_quartile,
        #               plot_joint_hist_kg_quartiles, plot_joint_future_kg_quartiles, plot_kg_score_quartile]

        # plot_list = [plot_future, plot_diff, plot_score,
        #               plot_joint_hist, plot_hist, plot_z,
        #                plot_joint_hist_quartiles, plot_joint_future_quartiles, plot_score_quartile]

        plot_list = plot_list + [plot_kg_score_quartile]

        # plot = hv.Layout(plot_list).cols(3)
        # hvplot.save(plot, f'{params["indicators"][ind][var]["short_name"]}_bivariate_dashboard_{str(gmt_threshold).replace(".", "p")}_interp.html')

    plot = (
        hv.Layout(plot_list)
        .cols(3)
        .opts(title=f'{str(gmt_threshold).replace(".", "p")}')
    )
    hvplot.save(
        plot,
        f'All_indicators_bivariate_dashboard_{str(gmt_threshold).replace(".", "p")}_interp.html',
    )

    # fig = plt.figure()
    # kg_hist_all.kg_class.plot(cmap='magma_r', vmin=0, vmax=6)
    # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_kg_quartiles_score_hist.png'))
    # plt.close()

    # fig = plt.figure()
    # kg_future_all.kg_class.plot(cmap='magma_r', vmin=0, vmax=6)
    # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_kg_quartiles_score_{str(gmt_threshold).replace(".", "p")}.png'))
    # plt.close()

    # Manual bins
    # hist_data_binned = bin_data(hist_data, [10, 50, 100])
    # future_data_binned = bin_data(future_data, [10, 50, 100])

    # fig = plt.figure()
    # hist_bivariate.plot(cmap='magma_r', vmin=0, vmax=6)
    # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_quartiles_score_hist.png'))
    # plt.close()

    # fig = plt.figure()
    # future_bivariate.plot(cmap='magma_r', vmin=0, vmax=6)
    # plt.savefig(os.path.join(plot_dir, f'{params["indicators"][ind][var]["short_name"]}_quartiles_score_{str(gmt_threshold).replace(".", "p")}.png'))
    # plt.close()

# temp4 = bin_data(t20_std_change, [1, 2, 3])
# temp4_abs = bin_data(t20_hist, [10, 50, 100])
# temp4_2p0 = bin_data(t20_2p0, [10, 50, 100])

# temp4 = bin_data(sdd_std_change, [1, 2, 3])
# temp4_abs = bin_data(sdd_hist, [250, 750, 1250])
# temp4_2p0 = bin_data(sdd_2p0, [250, 750, 1250])

# test = bin_data(pr20_std_change, [1, 2, 3])
# temp_abs = bin_data(pr20_hist, [5, 10, 20])
# temp4_2p0 = bin_data(pr20_2p0, [5, 10, 20])


#%%


dot = [3, 5, 7, 10]
quantiles = [0.95, 0.97, 0.99]
thresholds = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]


land_mask = xr.open_dataset("H:\\git\\climate_impacts_processing\\landareamaskmap0.nc")


output_dir = "H:\\hotspots_explorer\\outputs\\test_zeros\\multi-model"
os.chdir(output_dir)


hw_mm = xr.open_dataset(
    "H:/hotspots_explorer/outputs/test_zeros/multi-model/ISIMIP3b_MM_heatwave.nc4"
)
hw_diff = xr.open_dataset(
    "H:/hotspots_explorer/outputs/test_zeros/multi-model/ISIMIP3b_Diff_heatwave.nc4"
)
hw_scores = xr.open_dataset(
    "H:/hotspots_explorer/outputs/test_zeros/multi-model/ISIMIP3b_Scores_heatwave.nc4"
)


hw_mm = hw_mm.where(land_mask["land area"] > 0)


#%% STARTS here new

fn_ds = "C:\\Users\\byers\\IIASA\\ECE.prog - Documents\\Research Theme - NEXUS\\Hotspots_Explorer_2p0\\rcre_testing\\testing_2\\output\\maps\\"

ds = xr.open_dataset(fn_ds + "scenario_maps_multiindicator_score.nc")


#%%
plot_list = []
year = 2055
for v in ds.data_vars:

    new_plot = (
        ds[v].sel(year=year).hvplot(x="lon", y="lat", cmap="magma_r", shared_axes=True)
    )
    plot_list = plot_list + [new_plot]

plot = hv.Layout(plot_list).cols(3)
hvplot.save(plot, f"{fn_ds}_test_dashboard_score.html")


def plot_maps_dashboard(
    ds,
    filename=None,
    indicators=None,
    year=2050,
    cmap="magma_r",
    shared_axes=True,
    clim=None,
):

    # if indicators==None:
    # indicators = list(ds.data_vars)
    # elif isinstance(indicators, list):
    # if not all(x in ds.data_vars for x in indicators):
    # print('')
    # else:
    # try:
    # Your code here
    # except Exception as e:
    # print(f"Error: not all items in indicators were found in ds.")
    # elif not isinstance(indicators, list):
    # print('')
    # try:
    # nothing
    # except Exception as e:
    # print(f"Error: indicators must be of type list.")

    # Subset the dataset. Check dims and length

    ds = check_ds_dims(ds)

    # if 'year' in ds.dims:
    # ds = ds.sel(year=year)
    # elif len(ds.dims) != 2:
    # except Exception as e:
    # print(f"Error: Year not a dimension and more than 2 dimensions in dataset")

    ds = ds.sel(year=year)

    for i in indicators:

        new_plot = (
            ds[i]
            .sel(year=year)
            .hvplot(x="lon", y="lat", cmap="magma_r", shared_axes=True)
        )
        plot_list = plot_list + [new_plot]

    plot = hv.Layout(plot_list).cols(3)

    # Plot - check filename
    # if type(filename) is None:
    # filename = 'maps_dashboard_{model}_{scenario}.html'

    # elif (type(filename) is str):
    # if (filename[:-5]) != '.html':
    # except Exception as e:
    # print(f"filename {filename} must end with '.html'")
    # else:
    # except Exception as e:
    # print(f"filename must be string and end with '.html'")

    hvplot.save(plot, filename)


#%%
for q, dt in it.product(range(0, len(quantiles)), range(0, len(dot))):

    print(q, dt)

    plot_list = []

    for t in range(0, len(thresholds)):

        print(t)

        plot_mm = hw_mm.mean_dur[t, 1, :, :, q, dt].hvplot(
            x="lon", y="lat", clim=(0, 50), cmap="YlOrRd", shared_axes=False
        )
        plot_diff = hw_diff.mean_dur[t, 0, :, :, q, dt].hvplot(
            x="lon", y="lat", clim=(0, 500), cmap="Reds", shared_axes=False
        )
        plot_score = hw_scores.mean_dur[t, 0, :, :, q, dt].hvplot(
            x="lon", y="lat", clim=(0, 3), cmap="magma_r", shared_axes=False
        )
        plot_list = plot_list + [plot_mm, plot_diff, plot_score]

    plot = hv.Layout(plot_list).cols(3)
    hvplot.save(plot, f"hw_{str(quantiles[q])[2:]}_{dot[dt]}.html")
    del plot, plot_mm, plot_diff, plot_score, plot_list

#%%  plot function


#%%

plot_maps_dashboard(ds, year=2055)
os.startfile()
