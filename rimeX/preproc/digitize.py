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
# import xarray as xa

from rimeX.logs import logger, setup_logger, log_parser
from rimeX.config import CONFIG, config_parser
from rimeX.preproc.warminglevels import get_warming_level_file
from rimeX.preproc.regional_average import get_regional_averages_file

def load_seasonal_means_per_region(variable, model, experiment, region, subregion, weights, seasons=['annual', 'winter', 'spring', 'summer', 'autumn']):

    file = get_regional_averages_file(variable, model, experiment, region, weights)
    monthly = pd.read_csv(file, index_col=0)[subregion or region]
    ny = monthly.size // 12
    assert monthly.size == ny*12, "not all years have 12 months"
    matrix = monthly.values.reshape(ny, 12)

    seasonal_means = {}

    for season in seasons:
        month_indices = np.asarray(CONFIG["preprocessing.seasons"][season]) - 1
        seasonal_means[season] = matrix[:, month_indices].mean(axis=1)

    return pd.DataFrame(seasonal_means, index=[datetime.datetime.fromisoformat(ts).year for ts in monthly.index.values[11::12]])


def load_regional_indicator_data(variable, region, subregion, weights, season, models, experiments):
    """higher level function than load_seasonal_means_per_region

    => add historical data
    => add variable-specific processing (e.g. retrieve projection baseline)
    """
    all_data = {}
    for model in models:
        
        try:
            historical = load_seasonal_means_per_region(variable, model, "historical", region, subregion, weights, seasons=[season])[season]
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
                future = load_seasonal_means_per_region(variable, model, experiment, region, subregion, weights, seasons=[season])[season]

            except FileNotFoundError:
                logger.warning(f"=> file not Found")
                continue

            data = pd.concat([historical, future])

            if np.isnan(data.values).all(): 
                logger.warning(f"All NaNs: {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue

            elif np.isnan(data.values).any(): 
                logger.warning(f"Some NaNs ({np.isnan(data.values).sum()} out of {data.size}): {variable} | {model} | {region} | {subregion} | {weights} | {season}. Skip")
                continue
                # raise ValueError(f"{model} | {experiment} => some NaNs were found")

            # indicator-dependent treatment
            if variable in ("tas", "tasmin", "tasmax"):
                y1, y2 = CONFIG["preprocessing.projection_baseline"]
                data -= data.loc[y1:y2].mean()

            elif variable == "pr":
                y1, y2 = CONFIG["preprocessing.projection_baseline"]
                data = (data / data.loc[y1:y2].mean() - 1) * 100

            all_data[(model, experiment)] = data

    return all_data


def interpolate_warming_levels(impact_data_records, warming_level_step, by):
    # key = lambda r: (r.get('ssp_family'), r.get('year'), r['variable'], r.get('model'), r.get('scenario'))
    key_fn = lambda r: tuple(r.get(k, 0) for k in by)
    input_gwls = set(r['warming_level'] for r in impact_data_records)
    gwls = np.arange(min(input_gwls), max(input_gwls)+warming_level_step, warming_level_step)
    interpolated_records = []
    for key, group in groupby(sorted(impact_data_records, key=key_fn), key=key_fn):
        igwls, ivalues = np.array([(r['warming_level'], r['value']) for r in sorted(group, key=lambda r: r['warming_level'])]).T
        # assert len(igwls) == 6, f"Expected 6 warming level for {ssp_family},{year}. Got {len(igwls)}: {repr(igwls)}"
        values = np.interp(gwls, igwls, ivalues)
        # print(ssp_family, year, variable, igwls, ivalues, '=>', gwls, values)
        interpolated_records.extend([{"warming_level":wl, "value": v, **dict(zip(by, key)) } for wl, v in zip(gwls, values)])
    return interpolated_records


def interpolate_years(impact_data_records, years, by):
    interpolated_records = []
    key_fn = lambda r: tuple(r.get(k, 0) for k in by)
    for key, group in groupby(sorted(impact_data_records, key=key_fn), key=key_fn):
        group = sorted(group, key=lambda r: r['year'])
        iyears, ivalues = np.array([(r['year'], r['value']) for r in group]).T
        assert not (np.diff(iyears) == 0).any(), "some years were duplicate"
        # assert len(iyears) == 6, f"Expected 6 warming level for {scenario},{year}. Got {len(iyears)}: {repr(iyears)}"
        values = np.interp(years, iyears, ivalues)
        interpolated_records.extend([{"year": year, "value": value, **dict(zip(by, key))} for year, value in zip(years, values)])
    return interpolated_records


# Sort out the quantiles
QUANTILES_MAP = [(5, [" 5th", "|5th"]), (95, [" 95th","|95th"]), (50, ["median", "50th",""])]

def _sort_out_quantiles(sat_variables):
    """extract quantile map from table variables (used in `fit_records`)
    """
    sat_quantiles = {}
    sat_variables = list(sat_variables)
    for q, ss in QUANTILES_MAP:
        for v, s in itertools.product(list(sat_variables), ss):
            if s in v:
                sat_quantiles[q] = v
                sat_variables.remove(v)
                break
    assert not sat_variables and len(sat_quantiles) == 3
    return sat_quantiles


def fit_records(impact_data_records, samples, by, dist_name=None, sample_dim="sample"):
    """Expand a set of percentile records with proper samples (experimental)

    [
        {"variable": "...", "value": vmed, ...}, 
        {"variable": "...| 5th ...", "value": v5th, ...},
        {"variable": "...| 95th ...", "value": v95th, ...},
        ...
    ]

    with n samples

    [
        {"variable": "...", "value": v1}, 
        {"variable": "...", "value": v2},
        ...
        {"variable": "...", "value": vn},
        ...
    ]

    by fitting a distribution `dist` to each group (grouped using `by` parameter).

    This funciton is experimental because it works on somewhat subjective naming. 
    TODO: use the "quantile" field instead.
    """
    from rimeX.stats import fit_dist, repr_dist

    impact_variables = set(r['variable'] for r in impact_data_records)
    impact_years = sorted(set(r['year'] for r in impact_data_records))

    if len(impact_variables) != 3:
        logger.error(f"Expected three variables in impact fit mode. Found {len(impact_variables)}")
        logger.error(f"Remaining variable: {impact_variables}")
        raise ValueError

    try:
        sat_quantiles = _sort_out_quantiles(impact_variables)
    except Exception as error:
        logger.error(f"Failed to extract quantiles from impact table variables.")
        logger.error(f"Expected variables contained the following strings: {dict(QUANTILES_MAP)}")
        logger.error(f"Remaining variables: {str(impact_variables)}")
        raise

    key_fn = lambda r: tuple(r[f] for f in by)
    resampled_records = []
    # for tqdm... (distribution fitting takes time, not groupby, so OK)
    groupby_length = len(list(groupby(sorted(impact_data_records, key=key_fn), key=key_fn))) 

    for keys, group in tqdm.tqdm(groupby(sorted(impact_data_records, key=key_fn), key=key_fn), total=groupby_length):
        group = list(group)
        if len(group) != 3:
            logger.error(f'Expected group of 3 records (the percentiles). Got group of length {len(group)}:\n{keys}. Skip resampling.')
            logger.debug(f'{group}')
            resampled_records.append(group[0])
            continue
        by_var = {r['variable']: r for r in group}
        quants = [50, 5, 95]
        dist = fit_dist([by_var[sat_quantiles[q]]['value'] for q in quants], quants, dist_name=dist_name)
        logger.debug(f"{keys}: {repr_dist(dist)}")

        # resample (equally spaced percentiles)
        step = 1/samples
        values = dist.ppf(np.linspace(step/2, 1-step/2, samples))
        r0 = by_var[sat_quantiles[50]]
        for i, v in enumerate(values):
            resampled_records.append({**r0, **{"value": v, sample_dim: i}, **dict(zip(by, keys))})

    return resampled_records


def average_per_group(records, by, keep_meta=True):
    """(weighted) mean grouped by 

    Parameters
    ----------
    records: list of dict with keys "year", "value" [, "weight"] and elements of `by`
    by: list of keys for grouping

    Returns
    -------
    average_records: (weighted) list of records
    """
    average_records = []
    key_avg_year = lambda r: tuple(r.get(k, 0) for k in by)
    for key, group in groupby(sorted(records, key=key_avg_year), key=key_avg_year):
        if keep_meta: 
            group = list(group)
            first_record = group[0]
            meta = {k:v for k, v in first_record.items() if k not in ["value", "midyear", "weights"]}
        else:
            meta = {k:v for k,v in zip(by, key)}
        years, values, weights = np.array([[r["midyear"], r["value"], r.get("weight", 1)] for r in group]).T
        total_weight = weights.sum()
        mean_value = (values*weights).sum()/total_weight
        mean_year = (years*weights).sum()/total_weight
        mean_weight = (weights*weights).sum()/total_weight  

        average_records.append({"value": mean_value, "midyear": mean_year, "weight": mean_weight, **meta})

    return average_records


def make_equiprobable_groups(records, by):
    """Define a 'weight' field for each record to give equal weight for each model, if so required

    Parameters
    ----------
    records: a list of records with dict fields ["warming_level", "model"]
    by : group by keys
        e.g. ['warming_level', 'model']

    This function modifies the report in-place by defining a 'weight' field and does not return anything.

    Note
    ----
    This function ensure the sum of all weight per warming level is one.
    Note the normalization per warming level group is redundant with `recombine_magicc`, but we 
    do it anyway in case the functions are used independently, given the low computational cost of such an operation.
    """
    # key = lambda r: (r['warming_level'], r['model'])
    key_fn = lambda r: tuple(r.get(k, 0) for k in by)
    for r in records:
        r.setdefault("weight", 1)

    for key, group in groupby(sorted(records, key=key_fn), key=key_fn):
        group = list(group)

        # Make sure the weights are normalized within each temperature bin (because the number of models may vary)
        total_weights = sum(r["weight"] for r in group)

        for r in group:
            r['weight'] = r["weight"] / total_weights


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
            assert len(years) == 1, f"{wl}|{model}|{experiment} cannot have more than one year to load. Check that warming_level input file is correct."
            year = years[0]
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
    root=None, backend="csv"):

    if root is None: root = CONFIG["isimip.climate_impact_explorer"]
    if running_mean_window is None: running_mean_window = CONFIG["preprocessing.running_mean_window"]

    extensions = {
        "feather": ".ftr",
    }

    ext = extensions.get(backend, f".{backend}")

    scenarioavg = "_scenarioavg" if average_scenarios else ""
    othertags = ""
    if equiprobable_models:
        othertags = othertags + "_models-equi"
    return Path(root) / f"isimip_binned_data/{variable}/{region}/{subregion}/{weights}/{variable}_{region.lower()}_{subregion.lower()}_{season}_{weights.lower()}_{running_mean_window}-yrs{scenarioavg}{othertags}{ext}"


def get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season, overwrite=False, backends=["csv"], **kw):
    """ Same as bin_isimip_records but with cached I/O
    """
    supported_backend = ["csv", "feather", "parquet", "excel"]
    for backend in backends:
        if backend not in supported_backend:
            raise NotImplementedError(backend)

    binned_records_files = [get_binned_isimip_file(variable, region, subregion, weights, season, **kw, backend=backend) for backend in backends]

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

    indicator_data = load_regional_indicator_data(variable, region, subregion, weights, season, models, experiments)

    if len(indicator_data) == 0:
        logger.warning(f"No indicator data for {variable} | {region} | {subregion} | {weights} | {season}.")
        return []

    all_data = bin_isimip_records(indicator_data, warming_levels, meta={
        "region": region, 
        "subregion": subregion, 
        "weights": weights, 
        "season": season, 
        "variable": variable, 
        "unit":"",
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
    group.add_argument("-v", "--variable", nargs="+", choices=CONFIG["isimip.variables"], default=CONFIG["isimip.variables"])
    group.add_argument("--region", nargs="+", default=all_regions, choices=all_regions)
    group.add_argument("--all-subregions", action='store_true', help='include subregions as defined in CIE mask files')
    # group.add_argument("--subregion", nargs="+", help="if not provided, will default to region average")
    # group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", nargs="+", default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--season", nargs="+", default=list(CONFIG["preprocessing.seasons"]), choices=list(CONFIG["preprocessing.seasons"]))

    group = parser.add_argument_group('Result')
    group.add_argument("--backend", nargs="+", default=CONFIG["preprocessing.isimip_binned_backend"], choices=["csv", "feather"])
    group.add_argument("-O", "--overwrite", action='store_true')
    group.add_argument("--cpus", type=int)

    o = parser.parse_args()
    setup_logger(o)
        
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
            get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season, 
                running_mean_window=o.running_mean_window, overwrite=o.overwrite, backends=o.backend)

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
            jobs.append((executor.submit(get_binned_isimip_records, warming_levels, variable, region, subregion, weights, season, 
                running_mean_window=o.running_mean_window, overwrite=o.overwrite, backends=o.backend), (variable, region, subregion, weights, season)))
       
        # wait for the jobs to finish to exit this script
        for j, (job, ids) in enumerate(jobs):
            job.result()
            logger.info(f"Job {j+1} / {len(jobs)} completed: {' | '.join(ids)}")


if __name__ == "__main__":
    main()