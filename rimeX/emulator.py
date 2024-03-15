"""
For a given scenario, return the mapped percentile for an indicator
"""
from pathlib import Path
import argparse
import glob
import tqdm
from itertools import groupby
import numpy as np
import pandas as pd

from rimeX.logs import logger, log_parser
from rimeX.config import config, config_parser

from rimeX.warminglevels import get_warming_level_file
from rimeX.digitize import get_binned_isimip_records, make_models_equiprobable


def load_magicc_ensemble(file, projection_baseline=config['projection_baseline'], projection_baseline_offset=config['projection_baseline_offset']):
    """Read a MAGICC output file as a pandas DataFrame

    By default express w.r.t pre-industrial levels adjusted with observations around the projection baseline.
    """
    logger.info(f"Load MAGICC ensemble {file} with baseline {projection_baseline} and offset {projection_baseline_offset}")

    df = pd.read_csv(file, skiprows=23, sep="\s+", index_col=0)
    if projection_baseline is not None:
        y1, y2 = projection_baseline
        df -= df.loc[y1:y2].mean()

        if projection_baseline_offset is not None:
            df += projection_baseline_offset

    return df


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=True):
    """
    https://stackoverflow.com/a/75321415/2192272
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    i = np.argsort(values)
    sorted_weights = weights[i]
    sorted_values = values[i]
    Sn = np.cumsum(sorted_weights)

    if interpolate:
        Pn = (Sn - sorted_weights/2 ) / Sn[-1]
        return np.interp(quantiles, Pn, sorted_values)
    else:
        return sorted_values[np.searchsorted(Sn, np.asarray(quantiles) * Sn[-1])]



def deterministic_resampling(values, size, weights=None, rng=None, axis=None):
    """ Deterministic resampling of real-numbered values, with interpolation allowed
    """
    if rng is None:
        rng = np.random.default_rng()

    step = 1/size
    quantiles = np.linspace(step/2, 1-step/2, num=size)

    if weights is None:
        resampled = np.percentile(values, quantiles*100, axis=axis)

    else:
        resampled = weighted_quantiles(values, weights, quantiles)

    rng.shuffle(resampled)
    return resampled


def vectorize_impact_values(binned_isimip_data, samples, warming_levels, rng=None):
    impacts = np.empty(shape=(samples, warming_levels.size))
    impacts.fill(np.nan)

    # Vectorize impact values
    logger.info(f"Re-sample impact values (samples={samples})")
    key_wl = lambda r: r['warming_level']
    for wl, group in groupby(sorted(binned_isimip_data, key=key_wl), key=key_wl):
        i = np.searchsorted(warming_levels, wl)
        values, weights = np.array([[r['value'], r.get('weight', 1)] for r in group]).T
        weights /= weights.sum() # normalize weights within group
        # deterministic resampling and reshuffling
        impacts[:, i] = deterministic_resampling(values, size=samples, weights=weights, rng=rng)

    return impacts


def digitize_magicc(magicc_ensemble, warming_levels):
    logger.info(f"Digitize MAGICC values")
    bins = warming_levels[1:] - np.diff(all_warming_levels)/2
    return np.digitize(magicc_ensemble, bins)


def recombine_magicc_vectorized(binned_isimip_data, magicc_ensemble, quantile_levels=config["quantiles"], 
    samples=config.get('samples', 5000), seed=config.get('seed', None)):
    """Take binned ISIMIP data and MAGICC time-series as input and  returns quantiles as output

    This method uses Monte Carlo sampling.

    Parameters
    ----------
    binned_isimip_data : list of records with fields {"value": ..., "warming_level": ...}
    magicc_ensemble : pandas DataFrame with years as index and ensemble members as columns (warming since P.I.)
    quantile_levels : quantiles to include in the output, default from config.yml files

    Returns
    -------
    quantiles : pandas DataFrame (years as index, quantiles as columns)

    Note
    ----
    any weight normalization can be done prior to calling this function with define_weight_within_warming_levels
    """
    rng = np.random.default_rng(seed=seed)

    # bins for digitization
    warming_levels = np.sort(np.fromiter(set(r['warming_level'] for r in binned_isimip_data), float))

    impacts_resampled = vectorize_impact_values(binned_isimip_data, samples=samples, rng=rng, warming_levels=warming_levels)

    magicc_years = np.floor(magicc_ensemble.index.values).astype(int)
    magicc_ensemble = magicc_ensemble.values

    # resample MAGICC
    logger.info(f"Re-sample MAGICC values (samples={samples})")
    # resample_magicc_idx = rng.integers(magicc_ensemble.shape[1], size=samples)
    # magicc_ensemble = magicc_ensemble[:, resample_magicc_idx] # climate
    magicc_ensemble = deterministic_resampling(magicc_ensemble, size=samples, rng=rng, axis=1)

    # Digitize MAGICC
    # 0 means first warming level or less
    # bins.size = warming_level.size - 1  means last  warming level or more
    # bins can be irregularly spaced, that's OK (e.g. holes in the data)
    logger.info("Digitize MAGICC")
    bins = warming_levels[1:] - np.diff(warming_levels)/2  
    indices = np.digitize(magicc_ensemble, bins)

    allvalues = impacts_resampled[np.arange(samples), indices.T]

    badvalues = np.isnan(allvalues)
    if badvalues.any():
        logger.warning("Some NaNs found: intermediate ")

    quantiles = np.percentile(allvalues, np.array(quantile_levels)*100, axis=1).T
    return pd.DataFrame(quantiles, index=magicc_years, columns=quantile_levels)


def recombine_magicc(binned_isimip_data, magicc_ensemble, quantile_levels=config["quantiles"]):
    """Take binned ISIMIP data and MAGICC time-series as input and  returns quantiles as output

    Determinisitc method. This is the original method for "temperature" and "time" matching methods. 

    Parameters
    ----------
    binned_isimip_data : list of records with fields {"value": ..., "warming_level": ...}
    magicc_ensemble : pandas DataFrame with years as index and ensemble members as columns (warming since P.I.)
    quantile_levels : quantiles to include in the output, default from config.yml files

    Returns
    -------
    quantiles : pandas DataFrame (years as index, quantiles as columns)

    Note
    ----
    any weight normalization can be done prior to calling this function with define_weight_within_warming_levels
    """
    magicc_years = magicc_ensemble.index
    magicc_ensemble = magicc_ensemble.values

    # digitize MAGICC temperature
    # bins for digitization
    all_warming_levels = np.sort(np.fromiter(set(r['warming_level'] for r in binned_isimip_data), float))
    binsize = config["warming_level_step"] # this could possibly be derived from the above
    # assign any outlier to the edges, to keep the median unbiased
    bins = all_warming_levels[1:] - binsize/2
    indices = np.digitize(magicc_ensemble, bins)

    # Group data records by warming level
    key_wl = lambda r: r['warming_level']    
    binned_isimip_data_by_wl = {wl : list(group) for wl, group in groupby(sorted(binned_isimip_data, key=key_wl), key=key_wl)}

    # Now calculate quantiles for each year
    logger.info("Re-combine all data and calculate quantiles for each year")
    quantiles = np.empty((magicc_ensemble.shape[0], len(quantile_levels)))
                    
    for i, year in enumerate(tqdm.tqdm(magicc_years)):
        
        all_values = []
        all_weights = []

        # bincount: [0, 1, 0, 5, 2, 3, 0, 5] => [3, 1, 1, 1, 0, 2] (it count the occurences of w.l. indices: 3 x 0, 1 x 1, 1 x 2, 1 x 3, 0 x 4, 2 x 5)
        for idx, number_of_magicc_simulations in enumerate(np.bincount(indices[i])):
            # no need to calculate when no warming level bin is present
            if number_of_magicc_simulations == 0:
                continue  # i.e. idx = 0
            wl = all_warming_levels[idx]

            # probability p(GMT == wl)
            p_gmt = number_of_magicc_simulations / indices[i].size
            records = binned_isimip_data_by_wl[wl]
            
            values, weights = np.array([(r['value'], r.get('weight', 1)) for r in records]).T
            p_record = weights / weights.sum()

            all_values.append(values)
            all_weights.append(p_record * p_gmt)

        values = np.concatenate(all_values)
        weights = np.concatenate(all_weights)

        valid = np.isfinite(values)
        if (~valid).sum() > 0:
            logger.warning(f"{year}: {(~valid).sum()} invalid values out of {valid.size}")
        logger.debug(f"{year}: compute quantiles on {len(values[valid])} values")
        quantiles[i] = weighted_quantiles(values[valid], weights[valid], quantile_levels)

    return pd.DataFrame(quantiles, index=magicc_years.values.astype(int), columns=quantile_levels)


def main():

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser])
    
    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--matching-method", default=config['matching_method'])
    group.add_argument("--running-mean-window", default=config['running_mean_window'])
    group.add_argument("--warming-level-file", default=None)
    group.add_argument("--gmt-interannual-variability-sd", type=float, default=config['gmt_interannual_variability_sd'])
    group.add_argument("--samples", type=int, default=config['samples'])

    group = parser.add_argument_group('Indicator variable')
    group.add_argument("-v", "--variable", choices=config["variables"], required=True)
    group.add_argument("--region")
    group.add_argument("--subregion", help="if not provided, will default to region average")
    group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", required=True, choices=config["weights"])
    group.add_argument("--season", required=True, choices=list(config["seasons"]))

    # egroup = group.add_mutually_exclusive_group()
    # egroup.add_argument("--remove-baseline-temp", choices=config["variables"], required=True)

    group = parser.add_argument_group('Aggregation')
    group.add_argument("--individual-years", action="store_true")
    group.add_argument("--average-scenarios", action="store_true")
    group.add_argument("--equiprobable-models", action="store_true", help="if True, each model will have the same probability")
    group.add_argument("--model", nargs="+", help="if provided, only consider a set of specified model(s)")
    group.add_argument("--experiment", nargs="+", help="if provided, only consider a set of specified experiment(s)")
    group.add_argument("--quantiles", nargs='+', default=config['quantiles'])

    group = parser.add_argument_group('Scenario')
    group.add_argument("--magicc-files", nargs='+', required=True)

    group = parser.add_argument_group('Result')
    group.add_argument("-O", "--overwrite", action='store_true', help='overwrite final results')
    group.add_argument("--backend-isimip-bins", nargs="+", default=config["isimip_binned_backend"], choices=["csv", "feather"])
    parser.add_argument("--overwrite-isimip-bins", action='store_true', help='overwrite the intermediate calculations (binned isimip)')
    parser.add_argument("--overwrite-all", action='store_true', help='overwrite intermediate and final')
    group.add_argument("-o", "--output-file", required=True)

    o = parser.parse_args()

    if o.overwrite_all:
        o.overwrite = True
        o.overwrite_isimip_bins = True

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exist. Use -O or --overwrite to reprocess.")
        parse.exit(0)


    if o.region is None:
        all_regions = sorted([o.name for o in (Path(config["climate_impact_explorer_orig"])/"masks").glob("*")])
        print(f"--region required. Please choose a region from {', '.join(all_regions)}")
        parser.exit(1)
        return

    if o.list_subregions:
        import pickle
        pickle.load(open(f"{config['climate_impact_explorer_orig']}/masks/{o.region}/region_names.pkl", "rb"))
        print(f"All subregions for {o.region}: {', '.join(all_subregions)}")
        parser.exit(0)
        return

    if o.subregion is None:
        o.subregion = o.region
        
    if o.warming_level_file is None:
        o.warming_level_file = get_warming_level_file(**{**config, **vars(o)})

    if not Path(o.warming_level_file).exists():
        print(f"{o.warming_level_file} does not exist. Run warminglevels.py first.")
        parser.exit(1)
        return

    # Load Warming level table and bin ISIMIP data
    logger.info(f"Load warming level file {o.warming_level_file}")
    warming_levels = pd.read_csv(o.warming_level_file)

    binned_isimip_data = get_binned_isimip_records(warming_levels, o.variable, o.region, o.subregion, o.weights, o.season, 
        matching_method=o.matching_method, running_mean_window=o.running_mean_window, 
        individual_years=o.individual_years, average_scenarios=o.average_scenarios, 
        equiprobable_models=o.equiprobable_models,
        overwrite=o.overwrite_isimip_bins, backends=o.backend_isimip_bins)

    # Filter input data (experimental)
    if o.model is not None:
        binned_isimip_data = [r for r in binned_isimip_data if r['model'] in set(o.model)]
    if o.experiment is not None:
        binned_isimip_data = [r for r in binned_isimip_data if r['experiment'] in set(o.experiment)]

    # Load MAGICC data
    magicc_ensemble = []
    for file in o.magicc_files:
        magicc_ensemble.append(load_magicc_ensemble(file))
    magicc_ensemble = pd.concat(magicc_ensemble, axis=1)

    # Only use future values to avoid getting in trouble with the warming levels.
    magicc_ensemble = magicc_ensemble.loc[2015:]  

    assert np.isfinite(magicc_ensemble.values).all(), 'some NaN in MAGICC run'


    # Recombine MAGICC with binned ISIMIP data
    quantiles = recombine_magicc(binned_isimip_data, magicc_ensemble, o.quantiles)

    # Write result to disk
    logger.info(f"Write output to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    quantiles.to_csv(o.output_file)


if __name__ == "__main__":
    main()