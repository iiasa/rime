"""Preprocessing step. Followup of warminglevels.py to bin a given indicator variable into into warming level categories.

The script is optional. It provides preprocessing for all variables / region / season etc but could also be done on-the-fly in emulator.py (specific combination)
"""

from pathlib import Path
import datetime
import argparse
import glob
import tqdm
import concurrent.futures
import itertools
from itertools import groupby, product
import numpy as np
import pandas as pd
import xarray as xa

from rimeX.logs import logger, setup_logger, log_parser
from rimeX.config import CONFIG, config_parser
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory
from rimeX.preproc.regional_average import get_regional_averages_file, get_files
from rimeX.records import average_per_group, make_models_equiprobable

def load_seasonal_means_per_region(variable, model, experiment, region, subregion, weights, seasons=['annual', 'winter', 'spring', 'summer', 'autumn'], **kw):

    file = get_regional_averages_file(variable, model, experiment, region, weights, **kw)
    monthly = pd.read_csv(file, index_col=0)[subregion or region]

    if CONFIG.get(f"indicator.{variable}.frequency", "monthly") == "annual":
        seasonal_means = {"annual": monthly.values}
        years = monthly.index

    else:

        ny = monthly.size // 12
        assert monthly.size == ny*12, "not all years have 12 months"
        matrix = monthly.values.reshape(ny, 12)

        seasonal_means = {}

        for season in seasons:
            month_indices = np.asarray(CONFIG["preprocessing.seasons"][season]) - 1
            seasonal_means[season] = matrix[:, month_indices].mean(axis=1)

        years = np.array([datetime.datetime.fromisoformat(ts).year for ts in monthly.index.values[11::12]])

    return pd.DataFrame(seasonal_means, index=years)

def transform_indicator(data, indicator, meta=None):

    if meta is None:
        meta = CONFIG.get(f"indicator.{indicator}", {})

    y1, y2 = meta.get("projection_baseline", CONFIG["preprocessing.projection_baseline"])

    if meta.get("transform") == "baseline_change":
        data -= data.loc[y1:y2].mean()

    elif meta.get("transform") == "baseline_change_percent":
        data = (data / data.loc[y1:y2].mean() - 1) * 100

    elif meta.get("transform"):
        raise NotImplementedError(meta["transform"])

    return data


def load_regional_indicator_data(variable, region, subregion, weights, season, models, experiments, **kw):
    """higher level function than load_seasonal_means_per_region

    => add historical data
    => add variable-specific processing (e.g. retrieve projection baseline)
    """
    meta = CONFIG.get(f"indicator.{variable}", {})

    all_data = {}
    for model in models:

        if not meta.get("historical", True):
            # this is concatenated in the future scenario files
            historical = None

        else:
            try:
                historical = load_seasonal_means_per_region(variable, model, "historical", region, subregion, weights, seasons=[season], **kw)[season]
            except FileNotFoundError as error:
                logger.warning(str(error))
                logger.warning(f"=> Historical data file not found for {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue

            if np.isnan(historical).all():
                logger.warning(f"Historical is NaN for {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue

        for experiment in experiments:
            if experiment == "historical":
                continue

            logger.info(f"load {variable} | {model} | {experiment} | {region} | {subregion} | {weights} | {season}")

            try:
                future = load_seasonal_means_per_region(variable, model, experiment, region, subregion, weights, seasons=[season], **kw)[season]

            except FileNotFoundError:
                logger.warning(f"=> file not Found")
                continue

            if historical is not None:
                data = pd.concat([historical, future])
            else:
                data = future

            if np.isnan(data.values).all():
                logger.warning(f"All NaNs: {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue

            elif np.isnan(data.values).any():
                logger.warning(f"Some NaNs ({np.isnan(data.values).sum()} out of {data.size}): {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue
                # raise ValueError(f"{model} | {experiment} => some NaNs were found")

            # indicator-dependent treatment
            data = transform_indicator(data, variable, meta=meta)

            all_data[(model, experiment)] = data

    return all_data

def _bin_isimip_records(indicator_data, warming_levels,
    running_mean_window, warming_levels_reached=None, meta={}):
    """Load ISISMIP data for a {variable, region, subregion, weights, season}, binned according to warming levels.

    Parameters
    ----------
    warming_levels: pandas DataFrame (loaded from the warming level file)
        note it is expected to be consistent with the methods parameters
    variable, region, subregion, weights, season : ...
    running_mean_window : int, a number of year, default from config.toml file
    warming_levels_reached: set, optional
        by default all warming levels are used, but a subset can be provided to limit the calculations to fewer values actually used
    meta: dict, optional
        additional metadata to add in the records

    Returns
    -------
    binned_isimip_data: list of records with fields {"value": ..., "warming_level": ...} and more
    """
    binned_isimip_data = []

    key_wl = lambda r: r['warming_level']
    key_file = lambda r: (r['model'], r['experiment'])

    all_records = warming_levels.to_dict('records')

    if warming_levels_reached is None:
        warming_levels_reached = set(warming_levels['warming_level'].values)

    logger.info(f"load max {len(all_records)} records")

    # for (model, experiment), group in tqdm.tqdm(list(groupby(sorted(all_records, key=key_file), key=key_file))):
    for (model, experiment), group in groupby(sorted(all_records, key=key_file), key=key_file):

        group = list(group)
        assert len(group) > 0

        warming_level_in_this_group = set(r["warming_level"] for r in group)

        if set.isdisjoint(warming_levels_reached, warming_level_in_this_group):
            logger.warning(f"{model}|{experiment}: none of {len(warming_level_in_this_group)} warming levels are needed. Skip")
            continue

        if (model, experiment) not in indicator_data:
            logger.warning(f"{model}|{experiment} is not present in impact data: Skip")
            continue

        data = indicator_data[(model, experiment)]

        for wl, group2 in groupby(sorted(group, key=key_wl), key=key_wl):

            if wl not in warming_levels_reached:
                logger.info(f"{model}|{experiment}|{wl} not needed")
                continue

            years = [r['year'] for r in group2]

            # determine the year range to load
            # assert len(years) == 1, f"{wl}|{model}|{experiment} cannot have more than one year to load. Check that warming_level input file is correct."
            # year = years[0]
            for year in years:
                start, end = year - running_mean_window//2, year + running_mean_window//2

                datasel = data.loc[start:end]

                if np.isnan(datasel.values).any():
                    logger.warning(f"{model} | {experiment} | {start} to {end} => some NaNs were found")
                    if np.isnan(datasel.values).all():
                        raise ValueErrror(f"{model} | {experiment} => all NaNs slide")

                binned_isimip_data.append({"value":datasel.mean(axis=0), "model": model, "experiment": experiment, "year": year, "warming_level": wl, **(meta or {})})

    return binned_isimip_data


def bin_isimip_records(indicator_data, warming_levels,
    running_mean_window=None, average_scenarios=False, equiprobable_models=False,
    warming_levels_reached=None, meta=None):
    """Load ISISMIP data for a {variable, region, subregion, weights, season}, binned according to warming levels, and apply some additional filtering

    Parameters
    ----------
    indicator_data: isimip timeseries data (see load_regional_indicator_data)
    warming_levels: pandas DataFrame (loaded from the warming level file)
        note it is expected to be consistent with the methods parameters
    variable, region, subregion, weights, season : ...
    running_mean_window : int, a number of year, default from config.toml file
    average_scenarios : bool, False by default
        if True, average across scenarios (and years)
    equiprobable_models : bool, False by default
        if True, update weights to make each model equiprobable
    warming_levels_reached: set, optional
        by default all warming levels are used, but a subset can be provided to limit the calculations to fewer values actually used

    Returns
    -------
    binned_isimip_data: list of records with fields {"value": ..., "warming_level": ...} and more
    """
    if running_mean_window is None: running_mean_window = CONFIG["preprocessing.running_mean_window"]

    logger.info("bin ISIMIP data")
    binned_isimip_data = _bin_isimip_records(indicator_data, warming_levels,
        running_mean_window=running_mean_window, warming_levels_reached=warming_levels_reached, meta=meta)

    if average_scenarios:
        logger.info("average across scenarios (and years)")
        binned_isimip_data = average_per_group(binned_isimip_data, by=('model', 'warming_level'))

    # Harmonize weights
    if equiprobable_models:
        logger.info("Normalization to give equal weight for each model per temperature bin.")
        make_models_equiprobable(binned_isimip_data)

    if len(binned_isimip_data) == 0:
        raise RuntimeError("no data found !!")

    return binned_isimip_data


def get_binned_isimip_file(variable, region, subregion, weights, season,
    running_mean_window=None,
    average_scenarios=False,
    equiprobable_models=False,
    root=None, backend="csv", **kw):

    if root is None:
        root = get_root_directory(running_mean_window=running_mean_window, **kw)
    if running_mean_window is None: running_mean_window = CONFIG["preprocessing.running_mean_window"]

    extensions = {
        "feather": ".ftr",
    }

    ext = extensions.get(backend, f".{backend}")

    scenarioavg = "_scenarioavg" if average_scenarios else ""
    othertags = ""
    if equiprobable_models:
        othertags = othertags + "_models-equi"

    # if tuple(CONFIG['preprocessing.projection_baseline']) != (1995, 2014):
    # if "baseline" in CONFIG.get(f"indicator.{variable}.transform", "") and tuple(CONFIG['preprocessing.projection_baseline']) != (1995, 2014):
    if "baseline" in CONFIG.get(f"indicator.{variable}.transform", ""):
        y1, y2 = CONFIG.get(f"indicator.{variable}.projection_baseline", CONFIG['preprocessing.projection_baseline'])
        othertags = othertags + f"_baseline-{y1}-{y2}"

    return Path(root) / f"isimip_binned_data/{variable}/{region}/{subregion}/{weights}/{variable}_{region.lower()}_{subregion.lower()}_{season}_{weights.lower()}_{running_mean_window}-yrs{scenarioavg}{othertags}{ext}"


def get_indicator_units(variable, simulation_round=None):

    units = CONFIG.get(f"indicator.{variable}.units", "")
    if units:
        return units

    if CONFIG.get(f"indicator.{variable}.transform") == "baseline_change_percent":
        units = "%"
    else:
        file = get_files(variable, "*", "*", simulation_round=simulation_round)[0]
        logger.debug(f"Open {file} to find {variable} units")
        with xa.open_dataset(file) as ds:
            v = ds[variable]
            if hasattr(v, "units"):
                units = v.units
                logger.debug(f"{variable} units found: {units}")

            elif hasattr(v, "unit"):
                units = v.unit
                logger.debug(f"{variable} units found: {units}")

            else:
                logger.warning(f"Cannot find units for {variable}")
                units = ""

    return units


def get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season, overwrite=False, backends=["csv"], simulation_round=None, **kw):
    """ Same as bin_isimip_records but with cached I/O
    """
    supported_backend = ["csv", "feather", "parquet", "excel"]
    for backend in backends:
        if backend not in supported_backend:
            raise NotImplementedError(backend)

    binned_records_files = [get_binned_isimip_file(variable, region, subregion, weights, season, simulation_round=simulation_round, backend=backend, **kw) for backend in backends]

    for file, backend in zip(binned_records_files, backends):
        if not overwrite and file.exists():
            logger.info(f"Load binned ISIMIP data from {file}")
            if backend == "csv":
                df = pd.read_csv(file)
            elif backend == "feather":
                df = pd.read_feather(file)
            elif backend == "excel":
                df = pd.read_excel(file)
            elif backend == "parquet":
                df = pd.read_parquet(file)
            else:
                raise NotImplementedError(backend)
            return df.rename({"scenario":"experiment", "midyear": "year"}).to_dict("records")

    models = warming_levels['model'].unique().tolist()
    experiments = warming_levels['experiment'].unique().tolist()

    indicator_data = load_regional_indicator_data(variable, region, subregion, weights, season, models, experiments, simulation_round=simulation_round)

    if len(indicator_data) == 0:
        logger.warning(f"No indicator data for {variable} | {region} | {subregion} | {weights} | {season}.")
        return []

    units = get_indicator_units(variable, simulation_round=simulation_round)

    all_data = bin_isimip_records(indicator_data, warming_levels, meta={
        "region": region,
        "subregion": subregion,
        "weights": weights,
        "season": season,
        "variable": variable,
        "unit": units,
        }, **kw)

    for file, backend in zip(binned_records_files, backends):
        logger.info(f"Write binned data to {file}")
        file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_data).rename({"experiment":"scenario", "year":"midyear"}, axis=1)  # conform with pyam format
        if backend == "csv":
            df.to_csv(file, index=None)
        elif backend == "feather":
            df.to_feather(file)
        elif backend == "excel":
            df.to_excel(file)
        elif backend == "parquet":
            df.to_parquet(file)
        else:
            raise NotImplementedError(backend)

    return all_data


def get_subregions(region):
    import pickle
    file = Path(f'{CONFIG["preprocessing.regional.masks_folder"]}/{region}/region_names.pkl')
    if file.exists():
        pkl = pickle.load(open(file, "rb"))
        return list(pkl)
    else:
        return []
            # all_subregions = get_regional_averages_file(o.variable, o.model, o.experiment, o.region, o.weights).columns


def main():

    all_regions = sorted([f.name for f in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*") if (f/"masks").exists() and any((f/"masks").glob("*nc4"))])

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[config_parser, log_parser])

    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--running-mean-window", default=CONFIG["preprocessing.running_mean_window"], help="default: %(default)s years")
    group.add_argument("--warming-level-file", default=None)

    group = parser.add_argument_group('Indicator variable')
    group.add_argument("-v", "--variable", nargs="+", choices=CONFIG["indicators"], default=CONFIG["indicators"])
    group.add_argument("--region", nargs="+", default=all_regions, choices=all_regions)
    group.add_argument("--all-subregions", action='store_true', help='include subregions as defined in CIE mask files')
    # group.add_argument("--subregion", nargs="+", help="if not provided, will default to region average")
    # group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", nargs="+", default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--season", nargs="+", default=list(CONFIG["preprocessing.seasons"]), choices=list(CONFIG["preprocessing.seasons"]))
    group.add_argument("--simulation-round", default=CONFIG["isimip.simulation_round"], help="default: %(default)s")
    group.add_argument("--projection-baseline", default=CONFIG["preprocessing.projection_baseline"], type=int, nargs=2, help="default: %(default)s")

    group = parser.add_argument_group('Result')
    group.add_argument("--backend", nargs="+", default=CONFIG["preprocessing.isimip_binned_backend"], choices=["csv", "feather"])
    group.add_argument("-O", "--overwrite", action='store_true')
    group.add_argument("--cpus", type=int)

    o = parser.parse_args()
    setup_logger(o)

    CONFIG["isimip.simulation_round"] = o.simulation_round
    CONFIG["preprocessing.projection_baseline"] = o.projection_baseline

    if o.warming_level_file is None:
        o.warming_level_file = get_warming_level_file(**{**CONFIG, **vars(o)})

    if not Path(o.warming_level_file).exists():
        print(f"{o.warming_level_file} does not exist. Run warminglevels.py first.")
        parser.exit(1)
        return

    # Load Warming level table and bin ISIMIP data
    logger.info(f"Load warming level file {o.warming_level_file}")
    warming_levels = pd.read_csv(o.warming_level_file)

    all_items = [(variable, region, subregion, weights, season) for variable, region, weights, season in product(o.variable, o.region, o.weights, o.season) for subregion in [region]+(get_subregions(region) if o.all_subregions else [])]
    logger.info(f"Number of jobs (variables x region x subregion x weights x season): {len(all_items)}")

    if o.cpus is None or o.cpus < 2:

        for variable, region, subregion, weights, season in tqdm.tqdm(all_items):
            if CONFIG.get(f"indicator.{variable}.frequency", "monthly") == "annual" and season != "annual":
                continue
            get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season,
                running_mean_window=o.running_mean_window, overwrite=o.overwrite, backends=o.backend, simulation_round=o.simulation_round)

        parser.exit(0)


    ## parallel processing
    if o.cpus is not None:
        max_workers = min(o.cpus, len(all_items))
    else:
        max_workers = len(all_items)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = []

        logger.info(f"Digitize ISIMIP: Submit {len(all_items)} jobs.")
        for variable, region, subregion, weights, season in all_items:
            if CONFIG.get(f"indicator.{variable}.frequency", "monthly") == "annual" and season != "annual":
                continue
            jobs.append((executor.submit(get_binned_isimip_records, warming_levels, variable, region, subregion, weights, season,
                running_mean_window=o.running_mean_window, overwrite=o.overwrite, backends=o.backend, simulation_round=o.simulation_round), (variable, region, subregion, weights, season)))

        # wait for the jobs to finish to exit this script
        for j, (job, ids) in enumerate(jobs):
            job.result()
            logger.info(f"Job {j+1} / {len(jobs)} completed: {' | '.join(ids)}")


if __name__ == "__main__":
    main()