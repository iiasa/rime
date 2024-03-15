"""Preprocessing step. Followup of calculate_isimip_warming_levels.py to bin a given indicator variable into into warming level categories.

The script is optional. It provides preprocessing for all variables / region / season etc but could also be done on-the-fly in calculate_quantile_timeseries.py (specific combination)
"""

from pathlib import Path
import datetime
import argparse
import glob
import tqdm
import concurrent.futures
from itertools import groupby, product
import numpy as np
import pandas as pd
# import xarray as xa

from rimeX.logs import logger
from rimeX.config import config
from rimeX.calculate_isimip_warming_levels import get_warming_level_file
from rimeX.regional_average import get_regional_averages_file

def load_seasonal_means_per_region(variable, model, experiment, region, subregion, weights, seasons=['annual', 'winter', 'spring', 'summer', 'autumn']):

    file = get_regional_averages_file(variable, model, experiment, region, weights)
    monthly = pd.read_csv(file, index_col=0)[subregion]
    ny = monthly.size // 12
    assert monthly.size == ny*12, "not all years have 12 months"
    matrix = monthly.values.reshape(ny, 12)

    seasonal_means = {}

    for season in seasons:
        month_indices = np.asarray(config['seasons'][season]) - 1
        seasonal_means[season] = matrix[:, month_indices].mean(axis=1)

    return pd.DataFrame(seasonal_means, index=[datetime.datetime.fromisoformat(ts).year for ts in monthly.index.values[11::12]])


def load_regional_indicator_data(variable, region, subregion, weights, season='annual', models=config['models'], experiments=config['experiments']):
    """higher level function than load_seasonal_means_per_region

    => add historical data
    => add variable-specific processing (e.g. retrieve projection baseline)
    """
    all_data = {}
    for model in models:
        
        historical = load_seasonal_means_per_region(variable, model, "historical", region, subregion, weights, seasons=[season])[season]

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

            if np.isnan(data.values).any(): 
                raise ValueError(f"{model} | {experiment} => some NaNs were found")

            # indicator-dependent treatment
            if variable == "tas":
                y1, y2 = config["projection_baseline"]
                data -= data.loc[y1:y2].mean()

            elif variable == "pr":
                y1, y2 = config["projection_baseline"]
                data = (data / data.loc[y1:y2].mean() - 1) * 100

            all_data[(model, experiment)] = data

    return all_data


def resample_with_natural_variability(records,
    sigma=0.14,
    binsize=config["warming_level_step"],
    ):
    """Enlarge the warming_level: value mapping to account for natural variability

    Parameters
    ----------
    records: list of dict with at least the "warming_level" field
    sigma: standard deviation of interannual variability in GMT (default to 0.14)

    Returns
    -------
    extended_records: extended list of dict with "warming_level", "weight" and other fields 

    Note this function resets the `weight` attribute.
    """
    import math
    from scipy.stats import norm

    natvar_dist = norm(0, sigma)
    n = math.ceil(3 * sigma / binsize) # that's about 0.001 probability we cut-off on each side
    deltas_plus = np.arange(1, n+1)*binsize
    deltas_prob = natvar_dist.pdf(deltas_plus)/natvar_dist.pdf(0)

    extended_records = []
    key_wl = lambda r: r['warming_level']

    records_by_wl = {wl : list(group) for wl, group in groupby(sorted(records, key=key_wl), key=key_wl)}

    for wl in records_by_wl:

        # collect an enlarged collection, and weight according to the decreasing probability further away from MAGICC
        extended_records.extend(records_by_wl[wl])

        for delta, prob in zip(deltas_plus, deltas_prob):
            # keep the symmetry during sampling, to avoid biases
            if wl+delta not in records_by_wl or wl-delta not in records_by_wl:
                continue

            natvar_records = records_by_wl[wl+delta] + records_by_wl[wl-delta]

            extended_records.extend({**r, **{"warming_level": wl, "weight": prob/len(natvar_records)}} for r in natvar_records)

    return extended_records


def average_per_group(records, by):
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
    key_avg_year = lambda r: tuple(r[k] for k in by)
    for key, group in groupby(sorted(records, key=key_avg_year), key=key_avg_year):
        years, values, weights = np.array([[r["year"], r["value"], r.get("weight", 1)] for r in group]).T
        total_weight = weights.sum()
        mean_value = (values*weights).sum()/total_weight
        mean_year = (years*weights).sum()/total_weight
        mean_weight = (weights*weights).sum()/total_weight  

        # NOTE: the use case for the above is after resample_with_natural_variability, with individual_years = False
        average_records.append({"value": mean_value, "year": mean_year, "weight": mean_weight, **{k:v for k,v in zip(by, key)}})

    return average_records

def make_models_equiprobable(records):
    """Define a 'weight' field for each record to give equal weight for each model, if so required

    Parameters
    ----------
    records: a list of records with dict fields ["warming_level", "model"]
    equiprobable_models: if True, give equal weight for each model. Default is False.

    This function modifies the report in-place by defining a 'weight' field and does not return anything.

    Note
    ----
    This function ensure the sum of all weight per warming level is one.
    Note the normalization per warming level group is redundant with `recombine_magicc`, but we 
    do it anyway in case the functions are used independently, given the low computational cost of such an operation.
    """
    key = lambda r: (r['warming_level'], r['model'])
    for r in records:
        r.setdefault("weight", 1)

    for wl, group in groupby(sorted(records, key=key), key=key):
        group = list(group)

        # Make sure the weights are normalized within each temperature bin (because the number of models may vary)
        total_weights = sum(r["weight"] for r in group)

        for r in group:
            r['weight'] = r["weight"] / total_weights


def _bin_isimip_records(indicator_data, warming_levels, 
    matching_method=config["matching_method"], running_mean_window=config["running_mean_window"],
    warming_levels_reached=None):
    """Load ISISMIP data for a {variable, region, subregion, weights, season}, binned according to warming levels. 

    Parameters
    ----------
    warming_levels: pandas DataFrame (loaded from the warming level file)
        note it is expected to be consistent with the methods parameters
    variable, region, subregion, weights, season : ...
    matching_method : "time" or "temperature", default from config.yml file
        see documentation in calculate_warming_levels.py module
    running_mean_window : int, a number of year, default from config.yml file
    warming_levels_reached: set, optional
        by default all warming levels are used, but a subset can be provided to limit the calculations to fewer values actually used

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
            logger.warning(f"{model}|{experiment}: none of {len(warming_level_in_this_group)} warming levels are needed")
            continue

        data = indicator_data[(model, experiment)]

        for wl, group2 in groupby(sorted(group, key=key_wl), key=key_wl):

            if wl not in warming_levels_reached:
                logger.info(f"{model}|{experiment}|{wl} not needed")
                continue

            years = [r['year'] for r in group2]
            assert len(years) > 0

            # # determine the year range to load
            # if matching_method == "time":
            #     assert len(years) == 1, f"{wl}|{model}|{experiment} cannot have more than one year to load when matching GMT by {matching_method}. Check that warming_level input file is correct."
            #     year = years[0]
            #     start, end = year - running_mean_window//2, year + running_mean_window//2

            # elif matching_method == "temperature":
            #     if len(years)  == 1:
            #         logger.warning(f"{wl}|{model}|{experiment} only has 1 year, whereas > 1 would be expected with `{matching_method}` matching method. Check the warming level input file is correct.")
            #     start, end = min(years), max(years)
            # else:
            #     raise NotImplementedError(matching_method)

            # start, end = min(years), max(years)

            # # Select all years
            # if matching_method == "time":
            #     datasel = data.loc[start:end]

            # else:
            #     datasel = data.loc[np.array(years)]

            datasel = data.loc[np.array(years)]

            if np.isnan(datasel.values).any(): 
                raise ValueError(f"{model} | {experiment} => some NaNs were found")

            binned_isimip_data.extend(({"value":value, "model": model, "experiment": experiment, "year": year, "warming_level": wl} 
                for year, value in zip(datasel.index.values, datasel.values)))

    return binned_isimip_data


def bin_isimip_records(indicator_data, warming_levels, 
    matching_method=config["matching_method"], running_mean_window=config["running_mean_window"],
    individual_years=False, average_scenarios=False, equiprobable_models=False,
    gmt_interannual_variability_sd=config['gmt_interannual_variability_sd'], 
    warming_levels_reached=None):
    """Load ISISMIP data for a {variable, region, subregion, weights, season}, binned according to warming levels, and apply some additional filtering

    Parameters
    ----------
    warming_levels: pandas DataFrame (loaded from the warming level file)
        note it is expected to be consistent with the methods parameters
    variable, region, subregion, weights, season : ...
    matching_method : "time" or "temperature", default from config.yml file
        see documentation in calculate_warming_levels.py module
    running_mean_window : int, a number of year, default from config.yml file
    individual_years : bool, False by default
        if True, all years are included, thus including signal from natural variability. Otherwise only the climatological mean is included.
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

    # Load ISIMIP data
    logger.info("bin ISIMIP data")
    binned_isimip_data = _bin_isimip_records(indicator_data, warming_levels, 
        matching_method=matching_method, running_mean_window=running_mean_window, warming_levels_reached=warming_levels_reached)

    if matching_method == "pure":
        logger.info("resample with natural variability")
        binned_isimip_data = resample_with_natural_variability(binned_isimip_data, sigma=gmt_interannual_variability_sd)

    if not individual_years:
        logger.info("average across years")
        binned_isimip_data = average_per_group(binned_isimip_data, by=('model', 'warming_level', 'experiment'))

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
    matching_method=config["matching_method"], 
    running_mean_window=config["matching_method"], 
    individual_years=False, 
    average_scenarios=False, 
    equiprobable_models=False,
    gmt_interannual_variability_sd=config['gmt_interannual_variability_sd'], 
    root=config["climate_impact_explorer"], ext='.csv'):

    scenarioavg = "_scenarioavg" if average_scenarios else ""
    natvartag = f"natvar-sd-{gmt_interannual_variability_sd}" if matching_method == "pure" else f"_{running_mean_window}yrs_natvar{individual_years}"
    othertags = ""
    if individual_years:
        othertags = othertags + "_allyears"
    else:
        othertags = othertags + "_clim"
    if equiprobable_models:
        othertags = othertags + "_models-equi"
    return Path(root) / f"isimip_binned_data/{variable}/{region}/{subregion}/{weights}/{variable}_{region.lower()}_{subregion.lower()}_{season}_{weights.lower()}_by{matching_method}{natvartag}{scenarioavg}{othertags}{ext}"


def get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season, overwrite=False, backends=["csv"], **kw):
    """ Same as bin_isimip_records but with cached I/O
    """
    supported_backend = ["csv", "feather"]
    for backend in backends:
        if backend not in supported_backend:
            raise NotImplementedError(backend)

    binned_records_files = [get_binned_isimip_file(variable, region, subregion, weights, season, **kw, ext=f".{backend}") for backend in backends]

    for file, backend in zip(binned_records_files, backends):
        if not overwrite and file.exists():
            logger.info(f"Load binned ISIMIP data from {file}")
            if backend == "csv":
                df = pd.read_csv(file)
            elif backend == "feather":
                df = pd.read_feather(file)
            else:
                raise NotImplementedError(backend)
            return df.to_dict("records")

    indicator_data = load_regional_indicator_data(variable, region, subregion, weights, season)
    all_data = bin_isimip_records(indicator_data, warming_levels, **kw)

    for file, backend in zip(binned_records_files, backends):
        logger.info(f"Write binned data to {file}")
        file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_data)
        if backend == "csv":
            df.to_csv(file, index=None)
        elif backend == "feather":
            df.to_feather(file)
        else:
            raise NotImplementedError(backend)

    return all_data


def get_subregions(region):
    import pickle
    pkl = pickle.load(open(f"{config['climate_impact_explorer_orig']}/masks/{o.region}/region_names.pkl", "rb"))
    return pkl["all_subregions"]
        # all_subregions = get_regional_averages_file(o.variable, o.model, o.experiment, o.region, o.weights).columns


def main():

    all_regions = sorted([f.name for f in (Path(config["climate_impact_explorer_orig"])/"masks").glob("*")])    

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--matching-method", default=config['matching_method'])
    group.add_argument("--running-mean-window", default=config['running_mean_window'])
    group.add_argument("--warming-level-file", default=None)
    group.add_argument("--individual-years", action="store_true")
    group.add_argument("--average-scenarios", action="store_true")

    group = parser.add_argument_group('Indicator variable')
    group.add_argument("-v", "--variable", nargs="+", choices=config["variables"], default=config["variables"])
    group.add_argument("--region", nargs="+", default=all_regions, choices=all_regions)
    # group.add_argument("--subregion", nargs="+", help="if not provided, will default to region average")
    # group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", nargs="+", default=config["weights"], choices=config["weights"])
    group.add_argument("--season", nargs="+", default=list(config["seasons"]), choices=list(config["seasons"]))

    group = parser.add_argument_group('Result')
    group.add_argument("--backend", nargs="+", default=config["isimip_binned_backend"], choices=["csv", "feather"])
    group.add_argument("-O", "--overwrite", action='store_true')
    group.add_argument("--cpus", type=int)

    o = parser.parse_args()
        
    if o.warming_level_file is None:
        o.warming_level_file = get_warming_level_file(**{**config, **vars(o)})

    if not Path(o.warming_level_file).exists():
        print(f"{o.warming_level_file} does not exist. Run calculate_isimip_warming_levels.py first.")
        parser.exit(1)
        return

    # Load Warming level table and bin ISIMIP data
    logger.info(f"Load warming level file {o.warming_level_file}")
    warming_levels = pd.read_csv(o.warming_level_file)

    all_items = [(variable, region, subregion, weights, season) for variable, region, weights, season in product(o.variable, o.region, o.weights, o.season) for subregion in get_subregions(region)]
    logger.info(f"Number of jobs (variables x region x subregion x weights x season): {len(all_items)}")

    if o.cpus is None or o.cpus < 2:

        for variable, region, subregion, weights, season in tqdm.tqdm(all_items):
            get_binned_isimip_records(warming_levels, variable, region, subregion, weights, season, 
                o.matching_method, o.running_mean_window, o.individual_years, o.overwrite, o.backend)

        parse.exit(0)


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
                o.matching_method, o.running_mean_window, o.individual_years, o.overwrite, o.backend), (variable, region, subreion, weights, season)))
       
        # wait for the jobs to finish to exit this script
        for j, (job, ids) in enumerate(jobs):
            job.result()
            logger.info(f"Job {j+1} / {len(jobs)} completed: {' | '.join(ids)}")


if __name__ == "__main__":
    main()