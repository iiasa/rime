"""Quantile maps for impacts that do not stem from a regional average
"""

from itertools import groupby
import tqdm
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import xarray as xa

from rimeX.config import CONFIG, config_parser
from rimeX.logs import logger, log_parser
from rimeX.compat import open_mfdataset
from rimeX.stats import fast_quantile, fast_weighted_quantile
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory
from rimeX.preproc.digitize import transform_indicator

def make_quantile_map_array(indicator:Indicator, warming_levels:pd.DataFrame,
                            quantile_bins=10, season="annual", running_mean_window=21,
                            projection_baseline=None, equiprobable_models=False):

    simulations = indicator.simulations
    w = running_mean_window // 2
    quants = np.linspace(0, 1, quantile_bins)

    warming_level_data = []
    warming_level_coords = []

    keywl = lambda r: r["warming_level"]
    wl_records = sorted(warming_levels.to_dict(orient="records"), key=keywl)

    logger.info(f"Process quantile maps for {indicator.name} | {season}. Warming levels {wl_records[0]['warming_level']} to {wl_records[-1]['warming_level']}")

    for wl, group in groupby(wl_records, key=keywl):

        logger.info(f"==== {wl} ====")
        files_to_concat = []
        group = list(group)

        if equiprobable_models:
            # calculate how many times each model enters for each warming level, and later downweight it accordingly
            modelkey = lambda r: r["model"]
            model_frequencies = {k:len(list(g)) for k, g in groupby(sorted(group, key=modelkey), key=modelkey)}
            weights = []

        # for each warming level, we loop over the different climate models and climate scenarios and simulation years
        for r in tqdm.tqdm(group):

            # filter climate models and scenarios
            subsimus = [s for s in simulations if _matches(s["climate_scenario"], r["experiment"]) and _matches(s["climate_forcing"], r["model"])]
            if len(subsimus) == 0:
                logger.warning(f"No simulation for {indicator.name} {r['model']} {r['experiment']}: Skip")
                continue

            # some indicators have an additional "model" specifier (impact model)
            # here we further loop over that sub-group to make sure we match the correct historical and future simulations
            groupkey = lambda s: s.get("model") # None for indicators without model specifier

            for model, simus in groupby(sorted(subsimus, key=groupkey), key=groupkey):

                simus = list(simus)
                simus_historical = [s for s in simulations if _matches(s["climate_scenario"], "historical") and _matches(s["climate_forcing"], r["model"]) and _matches(s.get("model"), model)]

                assert len(simus_historical) == 1
                assert len(simus) == 1

                simu_historical = simus_historical[0]
                simu = simus[0]
                # warming_level	year
                filepath_hist = indicator.get_path(**simu_historical)
                filepath = indicator.get_path(**simu)

                with open_mfdataset([filepath_hist, filepath], combine='nested', concat_dim="time") as ds:

                    # only select relevant months
                    if season is not None:
                        season_mask = ds["time.month"].isin(CONFIG["preprocessing.seasons"][season])
                        seasonal_sel = ds[indicator.ncvar].isel(time=season_mask)

                    else:
                        seasonal_sel = ds[indicator.ncvar]

                    # mean over required time-slice
                    data = seasonal_sel.sel(time=slice(str(r['year']-w),str(r['year']+w))).mean("time").load()

                    # subtract the reference period or express as relative change
                    if indicator.transform:

                        # mean over the ref period
                        if projection_baseline is not None:
                            y1, y2 = projection_baseline
                            dataref = seasonal_sel.sel(time=slice(str(y1),str(y2))).mean("time").load()

                        data = transform_indicator(data, indicator.name, dataref=dataref)

                    # assign metadata
                    data = data.assign_coords({
                        "warming_level": r["warming_level"],
                        "model": r["model"],
                        "experiment": r["experiment"],
                        "midyear": r["year"],
                        })

                files_to_concat.append(data)

                # downweight models that are more frequent
                # NOTE we assume any impact model comes with the same frequency across climate models
                # i.e. we correct for the frequency of the climate models in the selection of the warming levels
                # but not in the occurence of impact models
                if equiprobable_models:
                    weights.append(1/model_frequencies[r["model"]])

        samples = xa.concat(files_to_concat, dim="sample")
        # quantiles = samples.quantile(quants, dim="sample")
        if equiprobable_models:
            quantiles = fast_weighted_quantile(samples, quants, weights=weights, dim="sample")
        else:
            quantiles = fast_quantile(samples, quants, dim="sample")
        warming_level_data.append(quantiles)
        warming_level_coords.append(wl)

    warming_level_data = xa.concat(warming_level_data, dim="warming_level")
    warming_level_data = warming_level_data.assign_coords({"warming_level": warming_level_coords}) # otherwise it's dropped apparently
    warming_level_data.name = indicator.name
    return warming_level_data


def get_filepath(name, season="annual", root_dir=None, suffix="", **kw):
    if root_dir is None:
        root_dir = get_root_directory(**kw)
    return root_dir / "quantilemaps" / name / f"{name}_{season}_quantilemaps{suffix}.nc"


def make_quantilemap_prediction(a, gmt, samples=100, seed=42, quantiles=[0.5, .05, .95]):
    """Make a prediction of the quantile map for a given global mean temperature

    Parameters
    ----------
    a : xa.DataArray as produced by make_quantile_map_array
    gmt : pandas DataFrame for the global mean temperature, with years as index and ensemble members as columns
    samples : number of samples to draw (default: 100)
    seed : random seed
    quantiles : quantiles to compute (default: [0.5, .05, .95])
        if None, all ensemble members are returned

    Returns
    -------
    sampled_maps : xa.DataArray with dimensions year, sample
    """
    interp = RegularGridInterpolator((a.warming_level.values, a.coords["quantile"].values), a.values, bounds_error=False)
    rng = np.random.default_rng(seed=seed)
    igmt = rng.integers(0, gmt.columns.size, size=samples)
    resampled_gmt = gmt.iloc[:, igmt]
    iquantiles = rng.integers(0, a.coords["quantile"].size, size=resampled_gmt.shape)
    sampled_maps = interp((resampled_gmt.values, a.coords["quantile"].values[iquantiles]))
    sampled_maps = xa.DataArray(sampled_maps, coords=[
        gmt.index, np.arange(samples), a.lat, a.lon], dims=["year", "sample", "lat", "lon"])
    if quantiles is not None:
        sampled_maps = fast_quantile(sampled_maps, quantiles, dim="sample")
    return sampled_maps


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[config_parser, log_parser])

    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--running-mean-window", default=CONFIG["preprocessing.running_mean_window"], help="default: %(default)s years")
    group.add_argument("--warming-level-file", default=None)
    group.add_argument("--warming-levels", type=float, default=CONFIG.get("preprocessing.quantilemap_warming_levels"), nargs='+', help="All warming levels by default")
    group.add_argument("--quantile-bins", default=CONFIG["preprocessing.quantilemap_quantile_bins"], type=int, help="default: %(default)s")
    group.add_argument("--equiprobable-climate-models", action='store_true', help="Downweight models that are more frequent in the warming level selection")

    group = parser.add_argument_group('Indicator variable')
    group.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    group.add_argument("-i", "--indicator", nargs='+', default=[], choices=CONFIG["indicator"], help="includes additional, secondary indicator with specific monthly statistics")
    group.add_argument("--season", nargs="+", default=list(CONFIG["preprocessing.seasons"]), choices=list(CONFIG["preprocessing.seasons"]))
    group.add_argument("--simulation-round", nargs="+", default=CONFIG["isimip.simulation_round"], help="default: %(default)s")
    group.add_argument("--projection-baseline", default=CONFIG["preprocessing.projection_baseline"], type=int, nargs=2, help="default: %(default)s")

    parser.add_argument("-O", "--overwrite", action='store_true')
    eg = parser.add_mutually_exclusive_group()
    eg.add_argument("--suffix", default="", help="add suffix to the output file name (to reflect different processing options)")
    eg.add_argument("--auto-suffix", action='store_true', help="add an automatically-generated suffix to the output file name (to reflect different processing options)")

    # group = parser.add_argument_group('Result')
    # group.add_argument("--backend", nargs="+", default=CONFIG["preprocessing.isimip_binned_backend"], choices=["csv", "feather"])
    # group.add_argument("-O", "--overwrite", action='store_true')
    # group.add_argument("--cpus", type=int)

    o = parser.parse_args()

    if o.auto_suffix:
        assert o.suffix == "", "Cannot use --suffix and --auto-suffix together"
        parts = []
        if o.running_mean_window != CONFIG["preprocessing.running_mean_window"]:
            parts.append(f"rmw{o.running_mean_window}")
        if o.warming_levels is not None:
            parts.append(f"wl{len(o.warming_levels)}")
        if o.quantile_bins != CONFIG["preprocessing.quantilemap_quantile_bins"]:
            parts.append(f"qb{o.quantile_bins}")
        if o.equiprobable_climate_models:
            parts.append("eq")
        o.suffix = "_" + "-".join(parts)

    CONFIG["isimip.simulation_round"] = o.simulation_round
    CONFIG["preprocessing.projection_baseline"] = o.projection_baseline

    if o.warming_level_file is None:
        o.warming_level_file = get_warming_level_file(**{**CONFIG, **vars(o)})

    warming_levels = pd.read_csv(o.warming_level_file)

    if o.warming_levels is not None:
        quantilemap_warming_levels = np.asarray(o.warming_levels)
        warming_levels = warming_levels[warming_levels["warming_level"].isin(quantilemap_warming_levels)]

    root_dir = Path(o.warming_level_file).parent


    for name in o.variable + o.indicator:
        indicator = Indicator.from_config(name)

        for season in o.season:
            if indicator.frequency == "annual" and season != "annual":
                continue
            filepath = get_filepath(indicator.name, season, root_dir=root_dir, suffix=o.suffix)
            if filepath.exists() and not o.overwrite:
                logger.info(f"{filepath} already exists. Use -O or --overwrite to reprocess.")
                continue
            filepath.parent.mkdir(parents=True, exist_ok=True)

            array = make_quantile_map_array(indicator,
                                            warming_levels,
                                            season=season,
                                            quantile_bins=o.quantile_bins,
                                            running_mean_window=o.running_mean_window,
                                            projection_baseline=o.projection_baseline,
                                            equiprobable_models=o.equiprobable_climate_models,
                                            )

            logger.info(f"Write to {filepath}")
            encoding = {array.name: {'zlib': True}}
            array.to_netcdf(filepath, encoding=encoding)


if __name__ == "__main__":
    main()