"""
For a given scenario, return the mapped percentile for an indicator
"""
from pathlib import Path
import argparse
import glob
import fnmatch
import itertools
import tqdm
from itertools import groupby
import numpy as np
import pandas as pd

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser

from rimeX.warminglevels import get_warming_level_file
from rimeX.digitize import get_binned_isimip_records, make_models_equiprobable


def load_magicc_ensemble(file, projection_baseline=None, projection_baseline_offset=None):
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


def digitize_gmt(gmt_ensemble, warming_levels):
    logger.info(f"Digitize GMT values")
    bins = warming_levels[1:] - np.diff(all_warming_levels)/2
    return np.digitize(gmt_ensemble, bins)


def recombine_gmt_vectorized(binned_isimip_data, gmt_ensemble, quantile_levels, samples=5000, seed=None):
    """Take binned ISIMIP data and GMT time-series as input and  returns quantiles as output

    This method uses Monte Carlo sampling.

    Parameters
    ----------
    binned_isimip_data : list of records with fields {"value": ..., "warming_level": ...}
    gmt_ensemble : pandas DataFrame with years as index and ensemble members as columns (warming since P.I.)
    quantile_levels : quantiles to include in the output, default from config.toml files

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

    gmt_years = np.floor(gmt_ensemble.index.values).astype(int)
    gmt_ensemble = gmt_ensemble.values

    # resample GMT
    logger.info(f"Re-sample GMT values (samples={samples})")
    # resample_gmt_idx = rng.integers(gmt_ensemble.shape[1], size=samples)
    # gmt_ensemble = gmt_ensemble[:, resample_gmt_idx] # climate
    gmt_ensemble = deterministic_resampling(gmt_ensemble, size=samples, rng=rng, axis=1)

    # Digitize GMT
    # 0 means first warming level or less
    # bins.size = warming_level.size - 1  means last  warming level or more
    # bins can be irregularly spaced, that's OK (e.g. holes in the data)
    logger.info("Digitize GMT")
    bins = warming_levels[1:] - np.diff(warming_levels)/2  
    indices = np.digitize(gmt_ensemble, bins)

    allvalues = impacts_resampled[np.arange(samples), indices.T]

    badvalues = np.isnan(allvalues)
    if badvalues.any():
        logger.warning("Some NaNs found: intermediate ")

    quantiles = np.percentile(allvalues, np.array(quantile_levels)*100, axis=1).T
    return pd.DataFrame(quantiles, index=gmt_years, columns=quantile_levels)


def recombine_gmt_ensemble(binned_isimip_data, gmt_ensemble, quantile_levels, match_year=False):
    """Take binned ISIMIP data and GMT time-series as input and  returns quantiles as output

    Determinisitc method. This is the original method for "temperature" and "time" matching methods. 

    Parameters
    ----------
    binned_isimip_data : list of records with fields {"value": ..., "warming_level": ...}
    gmt_ensemble : pandas DataFrame with years as index and ensemble members as columns (warming since P.I.)
    quantile_levels : quantiles to include in the output, default from config.toml files
    match_year : bool, False by default. 
        If True, the data will be grouped according to year as well as temperature.
        Some of the impact data has a "year" attribute for population growth scenario, which
        is not related to the year of the climate model time-series. The option is introduced for that situation.

    Returns
    -------
    quantiles : pandas DataFrame (years as index, quantiles as columns)

    Note
    ----
    any weight normalization can be done prior to calling this function with define_weight_within_warming_levels
    """
    gmt_years = gmt_ensemble.index
    gmt_ensemble = gmt_ensemble.values

    # digitize MAGICC temperature
    # bins for digitization
    all_warming_levels = np.sort(np.fromiter(set(r['warming_level'] for r in binned_isimip_data), float))
    binsize = all_warming_levels[1] - all_warming_levels[0]
    # assign any outlier to the edges, to keep the median unbiased
    bins = all_warming_levels[1:] - binsize/2
    indices = np.digitize(gmt_ensemble, bins)

    # Group data records by warming level
    if match_year:
        key_wl_year = lambda r: (r['warming_level'], r['year'])
        binned_isimip_data_by_wl_and_year = {(wl, year) : list(group) for (wl, year), group in groupby(sorted(binned_isimip_data, key=key_wl_year), key=key_wl_year)}
    else:
        key_wl = lambda r: r['warming_level']    
        binned_isimip_data_by_wl = {wl : list(group) for wl, group in groupby(sorted(binned_isimip_data, key=key_wl), key=key_wl)}

    # Now calculate quantiles for each year
    logger.info("Re-combine all data and calculate quantiles for each year")
    quantiles = np.empty((gmt_ensemble.shape[0], len(quantile_levels)))
                    
    for i, year in enumerate(tqdm.tqdm(gmt_years)):
        
        all_values = []
        all_weights = []

        # bincount: [0, 1, 0, 5, 2, 3, 0, 5] => [3, 1, 1, 1, 0, 2] (it count the occurences of w.l. indices: 3 x 0, 1 x 1, 1 x 2, 1 x 3, 0 x 4, 2 x 5)
        for idx, number_of_gmt_simulations in enumerate(np.bincount(indices[i])):
            # no need to calculate when no warming level bin is present
            if number_of_gmt_simulations == 0:
                continue  # i.e. idx = 0
            wl = all_warming_levels[idx]

            # probability p(GMT == wl)
            p_gmt = number_of_gmt_simulations / indices[i].size
            if match_year:
                records = binned_isimip_data_by_wl_and_year[(wl, year)]
            else:
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

    return pd.DataFrame(quantiles, index=pd.Index(gmt_years.values.astype(int), name='year'), columns=quantile_levels)


def validate_iam_filter(keyval):
    key, val = keyval.split("=")
    return key, val


# Sort out the quantiles
QUANTILES_MAP = [(5, [" 5th", "|5th"]), (95, [" 95th","|95th"]), (50, ["median", "50th",""])]

def _sort_out_quantiles(sat_variables):
    """extract quantile map from table variables
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


def main():

    from rimeX.datasets import get_datapath

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser])
    
    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--matching-method", default=CONFIG["emulator.experimental.matching_method"])
    group.add_argument("--running-mean-window", default=CONFIG["emulator.running_mean_window"])
    group.add_argument("--warming-level-file", default=None)

    group = parser.add_argument_group('Impact indicator')
    group.add_argument("-v", "--variable", nargs="*", required=True)
    group.add_argument("--region", required=True)
    group.add_argument("--format", default="ixmp4", choices=["ixmp4", "custom"])
    group.add_argument("--impact-file", nargs='+', default=[], 
        help=f'Files such as produced by Werning et al 2014 (.csv with ixmp4 standard). Also accepted is a glob * pattern to match downloaded datasets (see also rime-download-ls).')
    group.add_argument("--impact-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --impact-filter scenario='ssp2*'")

    group = parser.add_argument_group('Impact indicator (custom)')
    group.add_argument("--subregion", help="if not provided, will default to region average")
    group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", default='LonLatWeight', choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--season", default='annual', choices=list(CONFIG["preprocessing.seasons"]))

    group = parser.add_argument_group('Aggregation')
    group.add_argument("--individual-years", action="store_true")
    group.add_argument("--average-scenarios", action="store_true")
    group.add_argument("--equiprobable-models", action="store_true", help="if True, each model will have the same probability")
    group.add_argument("--model", nargs="+", help="if provided, only consider a set of specified model(s)")
    group.add_argument("--experiment", nargs="+", help="if provided, only consider a set of specified experiment(s)")
    group.add_argument("--quantiles", nargs='+', default=CONFIG["emulator.quantiles"])
    group.add_argument("--match-year-population", action="store_true")
    group.add_argument("--warming-level-step", default=CONFIG["emulator.warming_level_step"], type=float,
        help="Impact indicators will be interpolated to match this warming level")
    group.add_argument("--impact-fit", action="store_true", 
        help="""Fit a distribution to the impact data from which to resample. 
        Assumes the quantile variables are named "{NAME}|5th percentile" and "{NAME}|95th percentile".""")
    group.add_argument("--impact-dist", default="auto", 
        choices=["auto", "norm", "lognorm"], 
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--impact-samples", default=100, type=int, help="Number of samples to draw if --impact-fit is set")

    group = parser.add_argument_group('Scenario')
    group.add_argument("--iam-file", default=get_datapath("test_data/emissions_temp_AR6_small.xlsx"), help='pyam-readable data')
    group.add_argument("--iam-variable", default="*GSAT*", help="Filter iam variable")
    group.add_argument("--iam-scenario", help="Filter iam scenario e.g. --iam-scenario SSP1.26")
    group.add_argument("--iam-model", help="Filter iam model")
    group.add_argument("--iam-fit", action="store_true", help="Fit a distribution to GSAT from which to resample")
    group.add_argument("--iam-dist", default="auto", 
        choices=["auto", "norm", "lognorm"], 
        # choices=["auto", "norm", "lognorm"], 
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--iam-samples", default=100, type=int, help="GSAT samples to draw if --iam-fit is set")
    group.add_argument("--iam-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --iam model='IMAGE 3.0.1' scenario=SSP1.26")
    group.add_argument("--magicc-files", nargs='+', help='if provided these files will be used instead if iam scenario')
    group.add_argument("--projection-baseline", type=int, nargs=2, default=CONFIG['emulator.projection_baseline'])
    group.add_argument("--projection-baseline-offset", type=float, default=CONFIG['emulator.projection_baseline_offset'])
    group.add_argument("--time-step", type=int, help="GSAT time step. By default whatever time-step is present in the input file.")

    group = parser.add_argument_group('Result')
    group.add_argument("-O", "--overwrite", action='store_true', help='overwrite final results')
    group.add_argument("--backend-isimip-bins", nargs="+", default=CONFIG["emulator.isimip_binned_backend"], choices=["csv", "feather"])
    parser.add_argument("--overwrite-isimip-bins", action='store_true', help='overwrite the intermediate calculations (binned isimip)')
    parser.add_argument("--overwrite-all", action='store_true', help='overwrite intermediate and final')
    group.add_argument("-o", "--output-file", required=True)
    group.add_argument("--save-gsat", help='filename to save the processed GSAT (e.g. for debugging)')
    group.add_argument("--save-impact-table", help='file name to save the processed impacts table (e.g. for debugging)')

    o = parser.parse_args()

    setup_logger(o)

    if o.overwrite_all:
        o.overwrite = True
        o.overwrite_isimip_bins = True

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exist. Use -O or --overwrite to reprocess.")
        parser.exit(0)

    # Load GMT data
    if o.magicc_files:
        gmt_ensemble = []
        for file in o.magicc_files:
            gmt_ensemble.append(load_magicc_ensemble(file, o.projection_baseline, o.projection_baseline_offset))
        gmt_ensemble = pd.concat(gmt_ensemble, axis=1)

    else:
        if not o.iam_file:
            parser.error("Need to indicate MAGICC or IAM data file --iam-file")
            parser.exit(1)
            
        import pyam
        if not Path(o.iam_file).exists() and get_datapath(o.iam_file).exists():
            o.iam_file = str(get_datapath(o.iam_file))
        iamdf = pyam.IamDataFrame(o.iam_file)
        filter_kw = dict(o.iam_filter)
        if o.iam_variable: filter_kw['variable'] = o.iam_variable
        if o.iam_scenario: filter_kw['scenario'] = o.iam_scenario
        if o.iam_model: filter_kw['model'] = o.iam_model
        iamdf_filtered = iamdf.filter(**filter_kw)

        if len(iamdf_filtered) == 0:
            logger.error(f"0-length dataframe after applying filter: {repr(filter_kw)}")
            parser.exit(1)

        if len(iamdf_filtered.index) > 1:
            logger.error(f"More than one index after applying filter: {repr(filter_kw)}")
            logger.error(f"Remaining index: {str(iamdf_filtered.index)}")
            parser.exit(1)

        if not o.iam_fit and len(iamdf_filtered.variable) > 1:
            logger.error(f"More than one variable after applying filter: {repr(filter_kw)}")
            logger.error(f"Remaining variable: {str(iamdf_filtered.variable)}")
            parser.exit(1)

        if not o.iam_fit and len(iamdf_filtered) != len(iamdf_filtered.year):
            logger.error(f"More entries than years after applying filter: {repr(filter_kw)}")
            logger.error(f"E.g. entries for first year:\n{str(iamdf_filtered.filter(year=iamdf_filtered.year[0]).as_pandas())}")
            parser.exit(1)

        df = iamdf_filtered.as_pandas()

        if o.iam_fit:
            from rimeX.stats import fit_dist

            logger.info(f"Fit GSAT temperature distribution ({o.iam_dist}) with {o.iam_samples} samples.")

            if len(iamdf_filtered.variable) != 3:
                logger.error(f"Expected three variables in GSAT fit mode after applying filter: {repr(filter_kw)}. Found {len(iamdf_filtered.variable)}")
                logger.error(f"Remaining variable: {str(iamdf_filtered.variable)}")
                parser.exit(1)

            if len(iamdf_filtered) != len(iamdf_filtered.year)*3:
                logger.error(f"Number of entries expected: 3 * years after applying filter: {repr(filter_kw)}. Got {len(iamdf_filtered)} entries and {len(iamdf_filtered.year)} years.")
                logger.error(f"E.g. entries for first year:\n{str(iamdf_filtered.filter(year=iamdf_filtered.year[0]).as_pandas())}")
                parser.exit(1)                

            try:
                sat_quantiles = _sort_out_quantiles(iamdf_filtered.variable)
            except Exception as error:
                logger.error(f"Failed to extract quantiles.")
                logger.error(f"Expected variables contained the following strings: {dict(QUANTILES_MAP)}")
                logger.error(f"Remaining variables: {str(iamdf_filtered.variable)}")
                parser.exit(1)

            gmt_q = pd.DataFrame({q: df[df["variable"] == sat_quantiles[q]].set_index('year')['value'] for q in [50, 5, 95]})

            # Fit & resample
            nt = gmt_q.shape[0]
            ens = np.empty((nt, o.iam_samples))
            for i in range(nt):
                # fit
                quants = [50, 5, 95]
                dist = fit_dist(gmt_q.iloc[i][quants], quants, o.iam_dist)
                logger.debug(f"{i}: {dist.dist.name}({','.join([str(r) for r in dist.args])})")

                # resample (equally spaced percentiles)
                step = 1/o.iam_samples
                ens[i] = dist.ppf(np.linspace(step/2, 1-step/2, o.iam_samples))

            gmt_ensemble = pd.DataFrame(ens, index=gmt_q.index)

        else:
            gmt_ensemble = df.set_index('year')[['value']]

    if o.save_gsat:
        logger.info("Save GSAT...")        
        gmt_ensemble.to_csv(o.save_gsat)
        logger.info("Save GSAT...done")        


    if o.time_step:
        orig_time_step = gmt_ensemble.index[1] - gmt_ensemble.index[0]
        if o.time_step > orig_time_step and orig_time_step * (o.time_step//orig_time_step) == o.time_step:
            logger.info(f"Subsample GSAT to {o.time_step}-year(s) time-step")
            gmt_ensemble = gmt_ensemble.iloc[::o.time_step//orig_time_step]

        else:
            import xarray as xa
            logger.info(f"Interpolate GSAT to {o.time_step}-year(s) time-step...")
            years = np.arange(gmt_ensemble.index[0], gmt_ensemble.index[-1]+o.time_step, o.time_step)
            gmt_ensemble = xa.DataArray(gmt_ensemble.values, coords={"year": gmt_ensemble.index}, dims=['year', 'sample']).interp(year=years).to_pandas()
            logger.info(f"Interpolate GSAT to {o.time_step}-year(s) time-step...done")

    # IIASA format like Wernings et al 2024
    if o.impact_file:

        import pyam
        filtered_files = []
        for file in o.impact_file:
            # file can be provided directly
            if Path(file).exists():
                filtered_files.append(file)
            # Provided as data name (glob pattern) under rimeX_datasets
            else:
                for f in sorted(glob.glob(str(get_datapath(file)))):
                    filtered_files.append(f)

        if not len(filtered_files):
            logger.warn("Empty list of impact files.")
            print("See rime-download. E.g. rime-download --all")
            parser.exit(1)

        sep = '\n'
        logger.info(f"Load {len(filtered_files)} impact files")
        filter_kw = dict(o.impact_filter)
        filter_kw["variable"] = o.variable
        filter_kw["region"] = o.region
        impact_data_table = pyam.concat([pyam.IamDataFrame(f).filter(**filter_kw) for f in filtered_files])

        if len(impact_data_table.variable) == 0:
            logger.error(f"Empty climate impact file with variable: {o.variable} and region {o.region}")
            parser.exit(1)


        # Only handle one variable at a time
        if not o.impact_fit and len(impact_data_table.variable) > 1:
            print(f"More than one variable found.\n {sep.join(impact_data_table.variable)}\nPlease restrict the --variable filter.")
            parser.exit(1)


        # Only handle one region at a time
        if len(impact_data_table.region) > 1:
            print(f"More than one region found.\n {sep.join(impact_data_table.region)}\nPlease restrict the --region filter.")
            parser.exit(1)

        # Now convert into the records format 
        impact_data_frame = impact_data_table.as_pandas()
        impact_data_records = impact_data_frame.to_dict('records')

        # ADD 'warming_level' threshold if absent. For now assume scenarios like ssp1_2p0 ==> warming level = 2.0 
        for record in impact_data_records:
            if "warming_level" not in record:
                try:
                    ssp, gwl = record.pop("scenario").split("_")
                    record["warming_level"] = float(gwl.replace('p', '.'))
                    record["ssp_family"] = ssp
                except:
                    logger.error(f"Missing 'warming_level' field. Expected scenario such as ssp1_2p0 to derive warming_level. Got: {record.get('scenario')}")
                    raise

        # Interpolate records
        logger.info("Impact data: interpolate warming levels...")
        key = lambda r: (r['ssp_family'], r['year'], r['variable'])
        input_gwls = set(r['warming_level'] for r in impact_data_records)
        gwls = np.arange(min(input_gwls), max(input_gwls)+o.warming_level_step, o.warming_level_step)
        interpolated_records = []
        for (ssp_family, year, variable), group in groupby(sorted(impact_data_records, key=key), key=key):
            igwls, ivalues = np.array([(r['warming_level'], r['value']) for r in group]).T
            # assert len(igwls) == 6, f"Expected 6 warming level for {ssp_family},{year}. Got {len(igwls)}: {repr(igwls)}"
            values = np.interp(gwls, igwls, ivalues)
            interpolated_records.extend([{"warming_level":wl, "value": v, "year": year, "ssp_family": ssp_family, "variable": variable} for wl, v in zip(gwls, values)])
        # inplace operation
        impact_data_records = interpolated_records
        logger.info("Impact data: interpolate warming levels...done")

        # For population dataset the year can be matched to temperatrure time-series. It must be interpolated to yearly values first.
        if o.match_year_population:
            logger.info("Impact data: interpolate years...")
            interpolated_records = []
            key = lambda r: (r['ssp_family'], r['warming_level'], r['variable'])
            input_years = set(r['year'] for r in impact_data_records)
            years = gmt_ensemble.index
            for (ssp_family, wl, variable), group in groupby(sorted(impact_data_records, key=key), key=key):
                group = list(group)
                input_years = np.sort([r["year"] for r in group])
                iyears, ivalues = np.array([(r['year'], r['value']) for r in group]).T
                # assert len(iyears) == 6, f"Expected 6 warming level for {ssp_family},{year}. Got {len(iyears)}: {repr(iyears)}"
                values = np.interp(years, iyears, ivalues)
                interpolated_records.extend([{"warming_level":wl, "value": v, "year": year, "ssp_family": ssp_family, "variable": variable} for year, v in zip(years, values)])
            # inplace operation
            impact_data_records = interpolated_records
            logger.info("Impact data: interpolate years...done")


        # Fit and resample impact data if required
        if o.impact_fit:
            from rimeX.stats import fit_dist

            logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples.")

            impact_variables = set(r['variable'] for r in impact_data_records)
            impact_years = sorted(set(r['year'] for r in impact_data_records))

            if len(impact_variables) != 3:
                logger.error(f"Expected three variables in impact fit mode after applying filter: {repr(filter_kw)}. Found {len(impact_variables)}")
                logger.error(f"Remaining variable: {impact_variables}")
                parser.exit(1)           

            try:
                sat_quantiles = _sort_out_quantiles(impact_variables)
            except Exception as error:
                logger.error(f"Failed to extract quantiles from impact table variables.")
                logger.error(f"Expected variables contained the following strings: {dict(QUANTILES_MAP)}")
                logger.error(f"Remaining variables: {str(impact_variables)}")
                parser.exit(1)

            fields = ["warming_level", "year", "ssp_family"]
            key = lambda r: tuple(r[f] for f in fields)
            resampled_records = []
            for keys, group in groupby(sorted(impact_data_records, key=key), key=key):
                group = list(group)
                assert len(group) == 3, f'Expected group of 3 records (the percentiles). Got {len(group)}: {group}'
                by_var = {r['variable']: r for r in group}
                quants = [50, 5, 95]
                dist = fit_dist([by_var[sat_quantiles[q]]['value'] for q in quants], quants, dist_name=o.impact_dist)
                logger.debug(f"{keys}: {dist.dist.name}({','.join([str(r) for r in dist.args])})")

                # resample (equally spaced percentiles)
                step = 1/o.impact_samples
                values = dist.ppf(np.linspace(step/2, 1-step/2, o.impact_samples))
                r0 = by_var[sat_quantiles[50]]
                for v in values:
                    resampled_records.append({**r0, **{"value": v}})

            impact_data_records = resampled_records


        binned_impact_data = impact_data_records


    # CIE format as used in March 2024
    else:
        if o.region is None:
            all_regions = sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])
            print(f"--region required. Please choose a region from {', '.join(all_regions)}")
            parser.exit(1)
            return

        if o.list_subregions:
            import pickle
            pickle.load(open(f'{CONFIG["preprocessing.regional.masks_folder"]}/{o.region}/region_names.pkl', "rb"))
            print(f"All subregions for {o.region}: {', '.join(all_subregions)}")
            parser.exit(0)
            return

        if o.subregion is None:
            o.subregion = o.region
            
        if o.warming_level_file is None:
            o.warming_level_file = get_warming_level_file(**{**config, **vars(o)})

        if not Path(o.warming_level_file).exists():
            parser.error(f"{o.warming_level_file} does not exist. Run warminglevels.py first.")
            parser.exit(1)
            return

        # Load Warming level table and bin ISIMIP data
        logger.info(f"Load warming level file {o.warming_level_file}")
        warming_levels = pd.read_csv(o.warming_level_file)

        assert len(o.variable) == 1, "only one variable supported for CIE data"
        binned_impact_data = get_binned_isimip_records(warming_levels, o.variable[0], o.region, o.subregion, o.weights, o.season, 
            matching_method=o.matching_method, running_mean_window=o.running_mean_window, 
            individual_years=o.individual_years, average_scenarios=o.average_scenarios, 
            equiprobable_models=o.equiprobable_models,
            overwrite=o.overwrite_isimip_bins, backends=o.backend_isimip_bins)

        # Filter input data (experimental)
        if o.model is not None:
            binned_impact_data = [r for r in binned_impact_data if r['model'] in set(o.model)]
        if o.experiment is not None:
            binned_impact_data = [r for r in binned_impact_data if r['experiment'] in set(o.experiment)]

    if o.save_impact_table:
        logger.info("Save impact table...")
        pd.DataFrame(binned_impact_data).to_csv(o.save_impact_table, index=None)
        logger.info("Save impact table...done")


    # Only use future values to avoid getting in trouble with the warming levels.
    gmt_ensemble = gmt_ensemble.loc[2015:]  

    assert np.isfinite(gmt_ensemble.values).all(), 'some NaN in MAGICC run'

    # Recombine GMT ensemble with binned ISIMIP data
    quantiles = recombine_gmt_ensemble(binned_impact_data, gmt_ensemble, o.quantiles, match_year=o.match_year_population)

    # GMT result to disk
    logger.info(f"Write output to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    quantiles.to_csv(o.output_file)


if __name__ == "__main__":
    main()