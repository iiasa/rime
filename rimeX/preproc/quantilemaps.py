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
from rimeX.stats import fast_quantile, fast_weighted_quantile
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory, get_model_frequencies
from rimeX.preproc.digitize import transform_indicator
from rimeX.preproc.regional_average import get_all_regions, open_files

def catchwarnings(func):
    def wrapped(*args, **kwargs):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapped

@catchwarnings
def make_quantile_map_array(indicator:Indicator, warming_levels:pd.DataFrame,
                            quantile_bins=10, season="annual", running_mean_window=21,
                            projection_baseline=None, equiprobable_models=False,
                            skip_transform=False, open_func_kwargs={}):


    simulations = indicator.simulations
    w = running_mean_window // 2
    quants = np.linspace(0, 1, quantile_bins)

    warming_levels["model"] = warming_levels["model"].map(str.lower)
    warming_levels["experiment"] = warming_levels["experiment"].map(str.lower)
    warming_level_by_model_exp = warming_levels.set_index(["model", "experiment"])

    keywl = lambda r: r["warming_level"]
    wl_records = sorted(warming_levels.to_dict(orient="records"), key=keywl)
    logger.info(f"Collect quantile maps data for {indicator.name} | {season}. Warming levels {wl_records[0]['warming_level']} to {wl_records[-1]['warming_level']}")

    if equiprobable_models:
        model_frequencies = get_model_frequencies(warming_levels)

    collect = {}

    key_file = lambda r: (r["climate_forcing"], r["climate_scenario"], r.get("model"))

    for key, group in tqdm.tqdm(groupby(sorted(simulations, key=key_file), key=key_file), total=len(simulations)):
        group = list(group)
        assert len(group) == 1, group
        simu = group[0]
        if key[1] == "historical":
            continue # this will be covered by the projection
        simu_historical = {**simu, "climate_scenario": "historical"}

        # check if that simulaion is required
        try:
            extract = warming_level_by_model_exp.loc[key[0]].loc[key[1]][["year", "warming_level"]]
        except KeyError:
            logger.debug(f"No warming level calculation for {key[0]} {key[1]}: Skip")
            continue

        with open_files(indicator, [simu_historical, simu], **open_func_kwargs) as data:

            # only select relevant months
            if season is not None:
                season_mask = data["time.month"].isin(CONFIG["preprocessing.seasons"][season])
                seasonal_sel = data.isel(time=season_mask)

            else:
                seasonal_sel = data

            # subtract the reference period or express as relative change
            if indicator.transform and not skip_transform and projection_baseline is not None:
                y1, y2 = projection_baseline
                dataref = seasonal_sel.sel(time=slice(str(y1),str(y2))).mean("time").load()
                assert "time" not in dataref.dims, dataref.dims

            # make 21-year running mean
            data_smooth = seasonal_sel.rolling(time=running_mean_window, center=True).mean().load()

            # collect all time slices required for the warming levels
            assert len(extract) > 0, f"No warming levels for {key[0]} {key[1]}"

            for i in range(len(extract)):
                wl = extract.iloc[i]["warming_level"]
                year = int(extract.iloc[i]["year"])

                # mean over required time-slice
                # data = seasonal_sel.sel(time=slice(str(year-w),str(year+w))).mean("time").load()
                data = data_smooth.sel(time=str(year)).squeeze("time").load()

                # subtract the reference period or express as relative change
                if indicator.transform and not skip_transform:
                    data = transform_indicator(data, indicator.name, dataref=dataref).load()

                # assign metadata
                data = data.assign_coords({
                    "warming_level": wl,
                    "model": key[0],
                    "experiment": key[1],
                    "midyear": year,
                    })

                assert "time" not in data.dims, (data.dims, data.shape)

                # append the newly calculated data where it belongs
                values, weights = collect.setdefault(wl, ([], []))
                values.append(data)
                weights.append(1/model_frequencies[key[0]] if equiprobable_models else 1)


    logger.info(f"Compute quantiles for collected data {indicator.name} | {season}.")

    warming_level_coords = np.array(sorted(collect.keys()))

    # create an empty array to store the quantiles
    warming_level_data = xa.DataArray(np.empty((len(warming_level_coords), len(quants), *data.shape)),
                                      dims=["warming_level", "quantile", *data.dims],
                                      coords={
                                        "warming_level": warming_level_coords,
                                        "quantile": quants,
                                        **{k:data.coords[k] for k in data.dims},
                                        },
                                      name=indicator.name,
                                      )

    # now re-organize the collected values by warming level and calculate the quantiles
    for i,wl in enumerate(tqdm.tqdm(warming_level_coords)):

        values, weights = collect.pop(wl)

        samples = xa.concat(values, dim="sample")
        del values  # clear memory
        # quantiles = samples.quantile(quants, dim="sample")
        if equiprobable_models:
            quantiles = fast_weighted_quantile(samples, quants, weights=weights, dim="sample")
        else:
            quantiles = fast_quantile(samples, quants, dim="sample")

        warming_level_data.values[i] = quantiles.transpose("quantile", ...).values

        del samples # clear memory
        del quantiles # clear memory


    return warming_level_data


def chunked(dim, size, total_size):
    """
    Decorator to process the data in chunks
    (e.g. call quantile maps on 1 or 5 or 10 degrees latitude bands to reduce memory usage)
    """
    def decorator(func):
        def wrapped(indicator, warming_levels, open_func_kwargs={}, **kwargs):
            chunks = []
            for isel in range(0, total_size, size):
                logger.info(f"Chunk along {dim}: {isel} to {isel+size} of {total_size}")
                open_func_kwargs_ = {**open_func_kwargs, "isel": {dim: slice(isel, isel+size)}}
                result = func(indicator, warming_levels, open_func_kwargs=open_func_kwargs_, **kwargs)
                chunks.append(result)
            return xa.concat(chunks, dim=dim)
        return wrapped
    return decorator


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
    group.add_argument("--chunk-size", type=int, choices=[5, 10, 36, 60, 72, 90, 180], help="Process maps in smaller chunk to save memory usage (lat range = 360)")

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

    # to reduce the memory usage, it is possible to split the calls into smaller warming_levels chunks
    # and concat along the warming level dimension afterwards (it will be less efficient)
    if not o.regional and o.chunk_size is not None:
        make_quantile_map_array_ = chunked("lat", o.chunk_size, 360)(make_quantile_map_array)
    else:
        make_quantile_map_array_ = make_quantile_map_array

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

            array = make_quantile_map_array_(indicator,
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