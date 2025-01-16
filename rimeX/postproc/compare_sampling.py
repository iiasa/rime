import pandas as pd
import numpy as np
import xarray as xa
from scipy.io import loadmat
import matplotlib.pyplot as plt

from rimeX.datasets.manager import get_datapath
from rimeX.config import CONFIG
from rimeX.preproc.quantilemaps import make_quantilemap_prediction, get_filepath
from rimeX.preproc.digitize import get_binned_isimip_file
from rimeX.records import make_models_equiprobable
from rimeX.emulator import recombine_gmt_ensemble, recombine_gmt_vectorized


def _generate_data_from_records(indicator_name, gmt, region, subregion, season, weight, quantiles=[0.5, .05, .95], samples=[5000], clip=True, seed=42):
    fp = get_binned_isimip_file(indicator_name, region, subregion, weight, season)
    impact_data = pd.read_csv(fp)
    impact_data_records = impact_data.to_dict("records")
    make_models_equiprobable(impact_data_records)

    all_data = {}

    # counting things
    result_records_df = recombine_gmt_ensemble(impact_data_records, gmt, quantiles)
    result = result_records_df.to_xarray().to_array("quantile")
    all_data["records"] = result

    # now try the vectorized version (needs clipping GMT to be equivalent to above)
    if clip:
        gmt = gmt.clip(lower=impact_data["warming_level"].min(), upper=impact_data["warming_level"].max())

    for samples_ in samples:
        result = recombine_gmt_vectorized(impact_data_records, gmt, samples=samples_, seed=seed).quantile(quantiles, axis=1).T.to_xarray().to_array("quantile")
        all_data[f"records vectorized ({samples_})"] = result

    return xa.concat(list(all_data.values()), dim=pd.Index(list(all_data.keys()), name="sample"))


# .expand_dims(dim='sample').assign_coords(sample=["count all records"])


def generate_comparison_data(indicator_name, gmt, region, subregion, season, weight, samples=[5000], clip=True, suffix="_eq",
                             modes=["records", "deterministic", "factorial", "montecarlo"], **kw):

    all_data = {}

    if "records" in modes:
        all_data["records"] = _generate_data_from_records(indicator_name, gmt, region, subregion, season, weight, clip=clip, samples=samples)

    if not any(m in modes for m in ["deterministic", "factorial", "montecarlo"]):
        return

    fp = get_filepath(indicator_name, season=season, suffix=suffix, region=region, regional_weights=weight)

    with xa.open_dataset(fp) as ds:
        impact_data = ds[indicator_name].sel(region=subregion).load()

    samples = np.asarray(samples)

    if "deterministic" in modes:
        results = []
        for s in samples:
            result = make_quantilemap_prediction(impact_data, gmt, samples=s, quantiles=[0.5, .05, .95], clip=clip, **kw)
            results.append(result)

        results = xa.concat(results, dim=pd.Index(samples, name="sample"))

        all_data["deterministic"] = results

    if "factorial" in modes:
        results_factorial = make_quantilemap_prediction(impact_data, gmt, quantiles=[0.5, .05, .95], mode="factorial", clip=clip, **kw)
        results_factorial = results_factorial.expand_dims(dim='sample').assign_coords(sample=[impact_data["quantile"].size * gmt.shape[1]])
        all_data["factorial"] = results_factorial

    if "montecarlo" in modes:
        results_mc = []
        for s in samples:
            result = make_quantilemap_prediction(impact_data, gmt, samples=s, quantiles=[0.5, .05, .95], clip=clip, **kw, mode="montecarlo")
            results_mc.append(result)

        results_mc = xa.concat(results_mc, dim=pd.Index(samples, name="sample"))

        all_data["montecarlo"] = results_mc

    return all_data



def plot_comparison_data(all_data, suptitle="", ni=2, nj=2, figsize=(12, 8), ref_title="records", ref_label="records vectorized (5000)", fontsize="xx-small"):

    f, axes = plt.subplots(ni, nj, figsize=figsize)

    ref = all_data[ref_title].sel(sample=ref_label)

    # for i, (title, data) in enumerate([("deterministic", results), ("records", result_records), ("factorial", results_factorial), ("montecarlo", results_mc)]):
    for i, title in enumerate(all_data):
        data = all_data[title]
        ax = axes.flat[i]
        ax.set_title(title)
        for s in [None] + data.sample.values.tolist():
            if s is None:
                if title == ref_title:
                    continue
                result = ref
                # label = "deterministic 100000"
                label = ref_label
                kw = dict(color = "black", linewidth=0.5)
            else:
                result = data.sel(sample=s)
                label = s
                kw = dict()
            l, = ax.plot(result.year, result.sel(quantile=0.5), label=label, **kw)
            kw.pop('color', None)
            ax.plot(result.year, result.T.sel(quantile=[0.05, 0.95]), color=l.get_color(), linestyle="--", **kw)
        ax.legend(fontsize=fontsize)
        ax.grid()
        ax.set_xlim(1980, 2100)


    if suptitle:
        f.suptitle(suptitle)

    f.tight_layout()



from rimeX.cie import predict_from_records, predict_from_quantilemap
from rimeX.cie import load_magicc_mat

indicator_names = [
	"maize_yield",
	"rice_yield",
	"wheat_yield",
	"soy_yield",
	"daily_temperature_variability",
	"number_of_wet_days",
	"extreme_daily_rainfall",
	"rx5day",
	"wet_bulb_temperature",
	"river_discharge",
	# "runoff",
	"tas",
	"pr",
	# "tasmax",
	# "tasmin",
]


def plot_df(df, ax, color=None, label="", **kw):
    l, = ax.plot(df.index, df[0.5], color=color, label=label, **kw)
    # for k in [0.05, 0.95]:
    #     ax.plot(df[k], color=l.get_color(), linestyle="--", **kw)
    ax.fill_between(df.index, df[0.05], df[0.95], color=l.get_color(), alpha=0.2)


def plot_one(ax, magicc_scenario, indicator_name, region, subregion, season, weight, all_data=None, gmt=None, label=None, colors={}):

    if all_data is None:
        all_data = {}

    try:
        key = magicc_scenario, indicator_name, region, subregion, season, weight
        if key not in all_data:
            if gmt is None:
                gmt = load_magicc_mat(magicc_scenario)
            all_data[key] = predict_from_records(gmt, indicator_name, region, subregion, season, weight, samples=5000, vectorized=False)
        df = all_data[key]
        plot_df(df, ax, color=colors.get(label), label=label)

    except:
        print(f"Failed to process: {indicator_name}. Skip.")
        return

magicc_scenarios = ["RCP3PD", "RCP45", "RCP85"]
colors_ = ["tab:green", "tab:blue", "tab:orange"]
scenario_colors = dict(zip(magicc_scenarios, colors_))

seasons = ["annual", "winter", "spring", "summer", "autumn"]
_scolors = ["black", "tab:purple", "tab:green", "tab:orange", "tab:brown"]
season_colors = dict(zip(seasons, _scolors))


def plot_many_scenarios(magicc_scenarios, indicator_names, region, subregion, season, weight, gmts=None, all_data=None):

    if all_data is None:
        all_data = {}

    if gmts is None:
        gmts = {magicc_scenario: load_magicc_mat(magicc_scenario) for magicc_scenario in magicc_scenarios}

    f, axes = plt.subplots(4, 3, figsize=(12, 12))

    for i, indicator_name in enumerate(indicator_names):

        ax = axes.flat[i]

        for magicc_scenario in magicc_scenarios:
            gmt = gmts[magicc_scenario]

            plot_one(ax, magicc_scenario, indicator_name, region, subregion, season, weight, all_data=all_data, gmt=gmt, label=magicc_scenario, colors=scenario_colors)

        ax.legend()
        ax.grid()
        ax.set_title(f"{indicator_name}")

    f.suptitle(f"{region} | {subregion} | {season} | {weight}")
    f.tight_layout()

    return all_data


def plot_many_seasons(magicc_scenario, indicator_names, region, subregion, seasons, weight, all_data=None):

    if all_data is None:
        all_data = {}

    gmt = load_magicc_mat(magicc_scenario)

    f, axes = plt.subplots(4, 3, figsize=(12, 12))

    for i, indicator_name in enumerate(indicator_names):

        ax = axes.flat[i]

        for season in seasons:
            plot_one(ax, magicc_scenario, indicator_name, region, subregion, season, weight, all_data=all_data, gmt=gmt, label=season, colors=season_colors)

        ax.legend()
        ax.grid()
        ax.set_title(f"{indicator_name}")

    f.suptitle(f"{region} | {subregion} | {weight} | MAGICC {magicc_scenario}")
    f.tight_layout()

    return all_data


def plot_many_regions(magicc_scenario, indicator_names, regions, season, weight, all_data=None, subregions=None):

    if all_data is None:
        all_data = {}

    gmt = load_magicc_mat(magicc_scenario)

    f, axes = plt.subplots(4, 3, figsize=(12, 12))

    for i, indicator_name in enumerate(indicator_names):

        ax = axes.flat[i]

        for j, region in enumerate(regions):
            subregion = subregions[j] if subregions is not None else region
            plot_one(ax, magicc_scenario, indicator_name, region, subregion, season, weight, all_data=all_data, gmt=gmt, label=region)

        ax.legend()
        ax.grid()
        ax.set_title(f"{indicator_name}")

    f.suptitle(f"{subregion} | {season} | {weight} | MAGICC {magicc_scenario}")
    f.tight_layout()

    return all_data