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
from rimeX.compat import open_mfdataset, open_dataset
from rimeX.stats import fast_quantile, fast_weighted_quantile
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory
from rimeX.preproc.digitize import transform_indicator
from rimeX.preproc.regional_average import get_regional_averages_file, preload_masks_merged, calc_regional_averages, open_region_mask, get_all_regions


def open_map_files(indicator, simus):
    files = [indicator.get_path(**simu) for simu in simus]
    return open_mfdataset(files, combine='nested', concat_dim="time")[indicator.ncvar]

def get_all_subregion(region, weights="latWeight"):
    with open_region_mask(region, weights) as mask:
        return list(mask)

def _open_regional_data_from_csv(indicator, simu, regions, weights="latWeight", admin=True, **kwargs):
    """This function loads data from the CSV files
    """
    if regions is None:
        regions = get_all_regions()
    files = [get_regional_averages_file(indicator.name, simu["climate_forcing"], simu["climate_scenario"],
                                    region, weights, impact_model=simu.get("model"), **kwargs)
                                    for region in regions]

    n0 = len(files)

    missing_regions = [region for (f, region) in zip(files, regions) if not f.exists()]
    files = [f for f in files if f.exists()]
    if len(files) == 0:
        raise FileNotFoundError(f"No regional files found for {indicator.name} {simu['climate_forcing']} {simu['climate_scenario']} {simu.get('model')}")

    if len(files) < n0:
        logger.info(f"Missing regions {missing_regions}")
        logger.warning(f"Only {len(files)} out of {n0} files exist. Skip the missing ones.")

    if admin:
        dfs = pd.concat([pd.read_csv(file, index_col=0) for file in files], axis=1)  # concat region and their admin boundaries
    else:
        dfs = pd.concat([pd.read_csv(file, index_col=0).iloc[:, :1] for file in files], axis=1)  # only use the first column (full region)

    # make sure we have dates as index (and not just years, cause the calling function needs dates)
    dfs.index = pd.to_datetime(dfs.index.astype(str))

    return xa.DataArray(dfs,
        coords=[dfs.index, dfs.columns],
        dims=["time", "region"],
        name=indicator.ncvar,
        )

def get_merged_masks(regions, weights="latWeight", admin=True):
    key = (weights, len(regions) if len(regions) > 1 else regions[0], "admin" if admin else "noadmin")
    filepath = Path(CONFIG["indicators.folder"], "masks", "merged_"+"_".join(map(str, key))+".nc")
    if filepath.exists():
        logger.info(f"Load merged masks from {filepath}")
        return open_dataset(filepath)
    merged_masks = preload_masks_merged(regions, weights, admin)
    logger.info(f"Write merged masks to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    merged_masks.to_netcdf(filepath, encoding={v: {'zlib': True} for v in merged_masks.data_vars})
    return merged_masks


def _open_regional_data(indicator, simu, regions=None, weights="latWeight",
                        admin=True, save=True, load=True, load_csv=False, all_masks=None):
    """Load the gridded netCDF and compute the regional averages on the fly
    """
    file = indicator.get_path(**simu)
    file_regional = indicator.get_path(**simu, regional=True, regional_weight=weights)

    if load and file_regional.exists():
        logger.info(f"Load regional averages from {file_regional}")
        return open_dataset(file_regional)[indicator.ncvar]

    elif load_csv:
        logger.info(f"Load regional averages from CSV files")
        ds = _open_regional_data_from_csv(indicator, simu, regions, weights, admin)
        # if save:
        #     logger.info(f"Write regional averages to {file_regional}")
        #     ds.to_netcdf(file_regional, encoding={indicator.ncvar: {'zlib': True}})
        return ds

    if regions is None:
        regions = get_all_regions()

    if all_masks is None:
        all_masks = get_merged_masks(regions, weights, admin)

    with open_dataset(file) as ds:
        region_averages = calc_regional_averages(ds[indicator.ncvar], all_masks, name=indicator.ncvar)

    if save:
        logger.info(f"Write regional averages to {file_regional}")
        file_regional.parent.mkdir(parents=True, exist_ok=True)
        region_averages.to_netcdf(file_regional, encoding={indicator.ncvar: {'zlib': True}})

    return region_averages


def open_regional_files(indicator, simus, regions, weights, admin=True, **kwargs):
    if regions is None:
        regions = get_all_regions()
    all_masks = get_merged_masks(regions, weights, admin)
    return xa.concat([_open_regional_data(indicator, simu, regions=regions, weights=weights, admin=admin, all_masks=all_masks, **kwargs)
                      for simu in simus], dim="time") # historical and future


def open_files(indicator, simus, regional=False, **kwargs):
    if regional:
        return open_regional_files(indicator, simus, **kwargs)
    else:
        return open_map_files(indicator, simus)


def make_quantile_map_array(indicator:Indicator, warming_levels:pd.DataFrame,
                            quantile_bins=10, season="annual", running_mean_window=21,
                            projection_baseline=None, equiprobable_models=False,
                            skip_transform=False, open_func_kwargs={}):


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

                with open_files(indicator, [simu_historical, simu], **open_func_kwargs) as data:

                    # only select relevant months
                    if season is not None:
                        season_mask = data["time.month"].isin(CONFIG["preprocessing.seasons"][season])
                        seasonal_sel = data.isel(time=season_mask)

                    else:
                        seasonal_sel = data

                    # mean over required time-slice
                    data = seasonal_sel.sel(time=slice(str(r['year']-w),str(r['year']+w))).mean("time").load()

                    # subtract the reference period or express as relative change
                    if indicator.transform and not skip_transform:

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


def get_filepath(name, season="annual", root_dir=None, suffix="", regional=False, regional_weights="latWeight", **kw):
    if root_dir is None:
        root_dir = get_root_directory(**kw)
    if regional:
        return root_dir / "regional" / name / f"{name}_{season}_regional_{regional_weights}{suffix}.nc"
    else:
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
    group.add_argument("--skip-transform", action='store_true', help="Skip the transformation of the indicator (absolute indicator only)")
    group.add_argument("--regional", action='store_true', help="Process regional averages instead of lat/lon maps")

    group = parser.add_argument_group('Regional average variables')
    group.add_argument("--weight", default="latWeight", choices=CONFIG["preprocessing.regional.weights"], help="default: %(default)s")
    group.add_argument("--region", nargs="+", default=None, choices=get_all_regions(), help="Regions to process if --regional")
    group.add_argument("--no-save-region", action='store_false', dest="save_region", help="Do not save regional averages to disk")
    group.add_argument("--no-load-region", action='store_false', dest="load_region", help="Do not load regional averages from disk")
    # group.add_argument("--no-load-csv-region", action='store_false', dest="load_csv_region", help="Do not load regional averages from CSV files")
    group.add_argument("--load-csv-region", action='store_true', help="Load regional averages from CSV files")

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
        if o.skip_transform:
            parts.append("abs")
        if o.running_mean_window != CONFIG["preprocessing.running_mean_window"]:
            parts.append(f"rmw{o.running_mean_window}")
        if o.warming_levels is not None:
            parts.append(f"wl{len(o.warming_levels)}")
        if o.quantile_bins != CONFIG["preprocessing.quantilemap_quantile_bins"]:
            parts.append(f"qb{o.quantile_bins}")
        if o.equiprobable_climate_models:
            parts.append("eq")
        if o.regional and o.region is not None and o.region != get_all_regions():
            if len(o.region) == 1:
                parts.append(f"r{o.region[0]}")
            else:
                parts.append(f"r{len(o.region)}")
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
            filepath = get_filepath(indicator.name, season, root_dir=root_dir, suffix=o.suffix,
                                    regional=o.regional, regional_weights=o.weight)
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
                                            skip_transform=o.skip_transform,
                                            open_func_kwargs=dict(
                                                regional=o.regional,
                                                weights=o.weight,
                                                regions=o.region,
                                                save=o.save_region,
                                                load=o.load_region,
                                                load_csv=o.load_csv_region,
                                            ),
                                            )

            logger.info(f"Write to {filepath}")
            encoding = {array.name: {'zlib': True}}
            array.to_netcdf(filepath, encoding=encoding)


if __name__ == "__main__":
    main()