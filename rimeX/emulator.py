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
import xarray as xa

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser

from rimeX.preproc.warminglevels import get_warming_level_file
from rimeX.preproc.digitize import (
    get_binned_isimip_records, 
    make_equiprobable_groups, interpolate_years, interpolate_warming_levels, 
    fit_records)
from rimeX.compat import FastIamDataFrame, concat, read_table, _isnumerical
from rimeX.datasets import get_datapath


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


def recombine_gmt_ensemble(impact_data, gmt_ensemble, quantile_levels, match_year=False):
    """Take binned ISIMIP data and GMT time-series as input and  returns quantiles as output

    Determinisitc method. This is the original method. 

    Parameters
    ----------
    impact_data : pandas DataFrame or list of records with fields {"value": ..., "warming_level": ...}
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
    if isinstance(impact_data, pd.DataFrame):
        impact_data = impact_data.to_dict("records")

    gmt_years = gmt_ensemble.index
    gmt_ensemble = gmt_ensemble.values

    # digitize MAGICC temperature
    # bins for digitization
    all_warming_levels = np.sort(np.fromiter(set(r['warming_level'] for r in impact_data), float))
    binsize = all_warming_levels[1] - all_warming_levels[0]
    # assign any outlier to the edges, to keep the median unbiased
    bins = all_warming_levels[1:] - binsize/2
    indices = np.digitize(gmt_ensemble, bins)

    # Group data records by warming level
    if match_year:
        key_wl_year = lambda r: (r['warming_level'], r['year'])
        impact_data_by_wl_and_year = {(wl, year) : list(group) for (wl, year), group in groupby(sorted(impact_data, key=key_wl_year), key=key_wl_year)}
    else:
        key_wl = lambda r: r['warming_level']    
        impact_data_by_wl = {wl : list(group) for wl, group in groupby(sorted(impact_data, key=key_wl), key=key_wl)}

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
                records = impact_data_by_wl_and_year[(wl, year)]
            else:
                records = impact_data_by_wl[wl]
            
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



class ImpactDataInterpolator:
    """Interpolator class inspired from RegularGridInterpolator
    """
    def __init__(self, dataarray, **kwargs):

        if isinstance(dataarray, xa.Dataset):
            logger.debug("convert Dataset to DataArray")
            dataarray = dataarray.to_array("variable")
            logger.debug("convert Dataset to DataArray...done")

        logger.debug("rename dataarray")
        dataarray = dataarray.rename(get_rename_mapping(dataarray.dims))
        logger.debug("rename dataarray...done")
        assert "warming_level" in dataarray.dims

        logger.debug("check ssp_family")
        if "ssp_family" in dataarray.dims: 
            dataarray = dataarray.assign_coords({"ssp_family": _get_ssp_mapping(dataarray.ssp_family.values)})
        elif "scenario" in dataarray.dims:
            dataarray = dataarray.rename({"scenario": "ssp_family"}).assign_coords({"ssp_family": _get_ssp_mapping(dataarray.scenario.values)})
        logger.debug("check ssp_family...done")

        logger.debug("transpose dataarray")
        indices = [c for c in ["warming_level", "year", "ssp_family"] if c in dataarray.dims]
        dataarray = dataarray.transpose(*indices, ...)
        logger.debug("transpose dataarray...done")

        self.dataarray = dataarray
        self.kwargs = kwargs


    @classmethod
    def from_dataframe(cls, table, mapping=None, meta_levels=None, index_levels=None, **kwargs):

        if mapping:
            logger.debug("index rename")
            table = table.rename(mapping or {}, axis=1)
            logger.debug("index rename...done")

        # use ssp_family as grouping, if provided
        if "ssp_family" not in table.columns:
            logger.debug("check ssp_family")        
            if "scenario" in table.columns:
                # raise ValueError("Expected either ssp_family or scenario in the impact table")
                table["ssp_family"] = _get_ssp_mapping(table["scenario"].values)
            logger.debug("check ssp_family...done")        

        if index_levels is None:
            index_levels = [c for c in ['warming_level', 'year', 'ssp_family'] if c in table.columns]

        if meta_levels is None:
            meta_levels = [c for c in ['region', 'model', 'variable'] if c in table.columns]

        if "warming_level" not in table.columns:
            raise ValueError("impact table must contain `warming_level`")

        # Create a 2-D data frame indexed by year and warming level
        # (this is usually very fast)        
        logger.debug("reshape impact table with multi indices")
        levels = index_levels + meta_levels
        series = table.set_index(levels)['value'];

        if not series.index.is_unique:
            logger.warning("index is not unique: drop duplicates")
            series = table.drop_duplicates(levels).set_index(levels)['value']

        logger.debug("transform to xarray.DataArray")
        dataarray = xa.DataArray.from_series(series)

        return cls(dataarray, **kwargs)


    def hasyear(self):
        return "year" in self.dataarray.dims

    def hasssp(self):
        return "ssp_family" in self.dataarray.dims

    # def hasmultiplessp(self):
    #     return self.hasssp() and table["ssp_family"].unique().size > 1


    def __call__(self, values, **kwargs):
        return self.interpolate_scipy(values, **{**self.kwargs, **kwargs})


    def interpolate_scipy(self, gmt_table, method="linear", mapping=None, return_dataarray=False, ignore_year=False, ignore_ssp=False, bounds_error=False):

        from scipy.interpolate import RegularGridInterpolator

        logger.debug("rename gmt_table columns")
        gmt_table = gmt_table.rename({"value":"warming_level", **(mapping or {})}, axis=1)

        # try to derive the SSP family if present
        logger.debug("find out ssp_family")
        if "ssp_family" in gmt_table.columns:
            gmt_table["ssp_family"] = _get_ssp_mapping(gmt_table["ssp_family"].values)

        elif "scenario" in gmt_table.columns:
            gmt_scenario = gmt_table['scenario'].values
            try:
                gmt_table["ssp_family"] = _get_ssp_mapping(gmt_scenario)
            except:
                logger.debug("could not get the ssp_family from GMT scenario")

        logger.debug("find out ssp_family...done")

        gmt = gmt_table['warming_level'].values

        index_levels = ["warming_level"]
        meta_levels = [c for c in self.dataarray.dims if c not in ["warminglevel", "year", "ssp_family"]]

        logger.debug("check years")

        if self.hasyear():
            if ignore_year:
                meta_levels += ["year"]

            else:
                if "year" not in gmt_table.columns and gmt_table.index.name == "year":
                    gmt_table = gmt_table.reset_index()

                if "year" not in gmt_table.columns:
                    raise ValueError("Expected 'year' column in GMT input (because `year` is present in the impact table), but None was found. Set `ignore_year=True` to ignore the years (this will result in an outer product).")

                index_levels += ['year']

        logger.debug("check ssp_family")
        
        if self.hasssp():
            if ignore_ssp:
                meta_levels += ["ssp_family"]

            else:
                if "ssp_family" not in gmt_table.columns:
                    raise ValueError("Expected 'ssp_family' column in GMT input (because `ssp_family` is present in the impact table), but None was found. Set `ignore_ssp=True` to ignore the ssp_family (this will result in an outer product).")
                index_levels += ['ssp_family']

        logger.debug("build indices")

        index = []

        logger.debug("build warming_level index")
        index.append(gmt)

        if "year" in index_levels:
            gmt_year = gmt_table['year'].values
            index.append(gmt_year)

        if "ssp_family" in index_levels:
            logger.debug("build ssp_family index")
            gmt_ssp_family = gmt_table["ssp_family"].values
            index.append(gmt_ssp_family)

        indices = np.array(index).T

        interp = RegularGridInterpolator([self.dataarray.coords[k].values for k in index_levels], self.dataarray.transpose(*index_levels, ...).values, bounds_error=bounds_error, method=method)
        values = interp(indices, method=method)

        # first step build a self.dataarray
        logger.debug("rebuild a DataArray")
        other_dims = [d for d in self.dataarray.dims if d not in index_levels]
        data = xa.DataArray(values, dims=['index']+other_dims, coords={k:v for k, v in self.dataarray.coords.items() if k in other_dims})

        # ...also provide the detail of the multi-index
        midx = xa.Coordinates.from_pandas_multiindex(
            gmt_table.set_index([c for c in ["year", "scenario", "ssp_family", "warming_level"] if c in gmt_table.columns and c not in meta_levels]).index, 'index')
        data = data.assign_coords(midx)

        if return_dataarray:
            return data

        # ... now flatten to a Dataframe
        logger.debug("transform to DataFrame")
        return data.to_series().reset_index(name='value')


def recombine_gmt_table(impact_data, gmt, **kwargs):
    """this function aims to mimic Edward Byers' early table_impacts_gwl, which indexes the impact table 
    to provide a multi-indicator, multi-scenario emulated dataset, without accounting for uncertainties

    Parameters
    ----------
    impact_data: pandas DataFrame (or convertible to, e.g. list of dict) or xarray.DataArray
        Standard fields are:
            - "warming_level" (or "gwl" or "gmt") or similar (not case-sensitive)
            - "scenario" or "ssp_family"
            - "year": note this refers to the SSP year for population-aggregated data, not the original scenario time-series
            - "variable"
            - "model"
            - "region"
        To be compatible with Werning et al, a special case where "warming_level" is parsed from the scenario column is also supported.
        Scenario is of the form "ssp1_1p5".

    gmt: pandas DataFrame with columns ["year", "value", "scenario", "model"] 

    **kwargs: passed to ImpactDataInterpolator __call__ (`method`, `return_dataarray`, ...)


    Returns
    -------
    pandas DataFrame


    Notes
    -----
    The impact data's warming levels must be interpolated to the desired fine resolution before entering this function
    """
    # IAMDataFrame => DataFrame
    if hasattr(impact_data, "as_pandas"):
        impact_data = impact_data.as_pandas()

    if type(impact_data) is list:
        impact_data = pd.DataFrame(impact_data)

    if type(impact_data) is pd.DataFrame:
        interpolator = ImpactDataInterpolator.from_dataframe(impact_data)

    elif isinstance(impact_data, (xa.Dataset, xa.DataArray)):
        interpolator = ImpactDataInterpolator(impact_data)

    else:
        raise TypeError(f"Expeced list of dict, pandas.DataFrame, xarray.DataArray or xarray.Dataset, got: {type(impact_data)}")

    return interpolator(gmt, **kwargs)


def validate_iam_filter(keyval):
    key, val = keyval.split("=")
    try:
        val = int(val)
    except:
        try:
            val = float(val)
        except:
            pass
    return key, val


def _get_gmt_parser(ensemble=False):

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Scenario')
    group.add_argument("--gsat-file", default=get_datapath("test_data/emissions_temp_AR6_small.xlsx"), help='pyam-readable data')
    group.add_argument("--gsat-variable", default="*GSAT*", help="Filter iam variable")
    group.add_argument("--gsat-scenario", help="Filter iam scenario e.g. --gsat-scenario SSP1.26")
    group.add_argument("--gsat-model", help="Filter iam model")
    group.add_argument("--gsat-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --gsat model='IMAGE 3.0.1' scenario=SSP1.26")
    group.add_argument("--year", type=int, nargs="*", help="specify a set of years (e.g. for maps)")

    group.add_argument("--projection-baseline", type=int, nargs=2, default=CONFIG['emulator.projection_baseline'])
    group.add_argument("--projection-baseline-offset", type=float, default=CONFIG['emulator.projection_baseline_offset'])

    if ensemble:
        group.add_argument("--gsat-resample", action="store_true", help="Fit a distribution to GSAT from which to resample")
        group.add_argument("--gsat-dist", default="auto", 
            choices=["auto", "norm", "lognorm"], 
            # choices=["auto", "norm", "lognorm"], 
            help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
        group.add_argument("--gsat-samples", default=100, type=int, help="GSAT samples to draw if --gsat-fit is set")

        group.add_argument("--time-step", type=int, help="GSAT time step. By default whatever time-step is present in the input file.")
        group.add_argument("--save-gsat", help='filename to save the processed GSAT (e.g. for debugging)')

    if ensemble:
        group.add_argument("--magicc-files", nargs='+', help='if provided these files will be used instead if iam scenario')
    else:
        group.add_argument("--magicc-files", nargs='+', help=argparse.SUPPRESS)

    if ensemble:
        group.add_argument("--no-check-single-index", action='store_false', dest='check_single_index', help=argparse.SUPPRESS)
    else:
        group.add_argument("--check-single-index", action='store_true', help=argparse.SUPPRESS)
        
    return parser



def _get_gmt_dataframe(o, parser):

    assert not o.magicc_files

    if not o.gsat_file:
        parser.error("Need to indicate MAGICC or IAM data file --gsat-file")
        parser.exit(1)
        
    if not Path(o.gsat_file).exists() and get_datapath(o.gsat_file).exists():
        o.gsat_file = str(get_datapath(o.gsat_file))

    df_wide = read_table(o.gsat_file)

    if o.pyam:
        import pyam
        iamdf = pyam.IamDataFrame(df_wide)
    else:
        iamdf = FastIamDataFrame(df_wide)

    filter_kw = {}
    for k, v in o.gsat_filter:
        filter_kw.setdefault(k, [])
        filter_kw[k].append(v)

    if o.gsat_variable: filter_kw['variable'] = o.gsat_variable
    if o.gsat_scenario: filter_kw['scenario'] = o.gsat_scenario
    if o.gsat_model: filter_kw['model'] = o.gsat_model
    iamdf_filtered = iamdf.filter(**filter_kw)

    if len(iamdf_filtered) == 0:
        logger.error(f"0-length dataframe after applying filter: {repr(filter_kw)}")
        parser.exit(1)

    if o.check_single_index:
        if len(iamdf_filtered.index) > 1:
            logger.error(f"More than one index after applying filter: {repr(filter_kw)}")
            logger.error(f"Remaining index: {str(iamdf_filtered.index)}")
            parser.exit(1)

        if not o.gsat_resample and len(iamdf_filtered.variable) > 1:
            logger.error(f"More than one variable after applying filter: {repr(filter_kw)}")
            logger.error(f"Remaining variable: {str(iamdf_filtered.variable)}")
            parser.exit(1)

        if not o.gsat_resample and len(iamdf_filtered) != len(iamdf_filtered.year):
            logger.error(f"More entries than years after applying filter: {repr(filter_kw)}. Years: {len(iamdf_filtered.year)}. Entries: {len(iamdf_filtered)}")
            logger.error(f"E.g. entries for first year:\n{str(iamdf_filtered.filter(year=iamdf_filtered.year[0]).as_pandas())}")
            parser.exit(1)

    df = iamdf_filtered.as_pandas()
    df2 = df.drop_duplicates()
    if len(df2) < len(df):
        logger.warning(f"Drop duplicates: GMT size {len(df)} => {len(df2)}")
        df = df2
    return df


def _get_gmt_ensemble(o, parser):

    if o.magicc_files:
        gmt_ensemble = []
        for file in o.magicc_files:
            gmt_ensemble.append(load_magicc_ensemble(file, o.projection_baseline, o.projection_baseline_offset))
        gmt_ensemble = pd.concat(gmt_ensemble, axis=1)

    else:
        df = _get_gmt_dataframe(o, parser)

        if o.gsat_resample:
            from rimeX.preproc.digitize import QUANTILES_MAP, _sort_out_quantiles
            from rimeX.stats import fit_dist

            logger.info(f"Fit GSAT temperature distribution ({o.gsat_dist}) with {o.gsat_samples} samples.")

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
            ens = np.empty((nt, o.gsat_samples))
            for i in range(nt):
                # fit
                quants = [50, 5, 95]
                dist = fit_dist(gmt_q.iloc[i][quants], quants, o.gsat_dist)
                logger.debug(f"{i}: {dist.dist.name}({','.join([str(r) for r in dist.args])})")

                # resample (equally spaced percentiles)
                step = 1/o.gsat_samples
                ens[i] = dist.ppf(np.linspace(step/2, 1-step/2, o.gsat_samples))

            gmt_ensemble = pd.DataFrame(ens, index=gmt_q.index)

        else:
            gmt_ensemble = df.set_index('year')[['value']]


    if o.year is not None:
        gmt_ensemble = gmt_ensemble.loc[o.year]

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


    if o.save_gsat:
        logger.info("Save GSAT...")
        gmt_ensemble.to_csv(o.save_gsat)
        logger.info("Save GSAT...done")

    return gmt_ensemble


def _simplify(name):
    name = name.replace("_","").replace("-","").lower()
    if name in ("warminglevel", "gwl", "gmt"):
        name = "warming_level"
    elif name in ("sspfamily", "ssp"):
        name = "ssp_family"
    elif name in ("experiment"):
        name = "scenario"
    return name


def get_rename_mapping(names):
    simplified = [_simplify(nm) for nm in names]
    if len(set(simplified)) != len(names):
        logger.error(f"input names: {names}")
        logger.error(f"would be renamed to: {simplified}")
        raise ValueError("some column names are duplicate or ambiguous")

    return dict(zip(names, simplified))


def _get_ssp_mapping(scenarios):
    """Returns a common mapping for SSP family, in the form of ["ssp1", etc..]
    """
    if _isnumerical(scenarios[0]):
        # return [f"ssp{int(s)}" for s in scenarios]
        return scenarios
    else:
        return [int(s[3]) for s in scenarios]


def _parse_warming_level_and_ssp(scenarios):
    """ Parse warming levels and SSP family from the Werninge et al scenarios

    Parameters
    ----------
    scenarios: array-like of type "ssp1_2p5"

    Returns
    -------
    warming_levels: float, array-like (global warming levels)
    scenarios: list of strings ["ssp1", ...]
    """
    warming_levels = np.empty(len(scenarios))
    ssp_family = []
    try:
        for i, value in enumerate(scenarios):
            ssp, gwl = value.split("_")
            warming_levels[i] = float(gwl.replace('p', '.'))
            ssp_family.append(ssp)
    except:
        logger.error(f"Expected scenario such as ssp1_2p0 to derive warming_level. Got: {value}")
        raise

    return warming_levels, ssp_family


def homogenize_table_names(df):
    """Make sure loosely named input table names are understood by the script, e.g. WarmingLevel or gwl or gmt => warming_level,
    and retrieve the warming level from the scenario name if need be.

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    DataFrame
    """
    names = df.columns

    mapping = get_rename_mapping(names)

    df = df.rename(mapping, axis=1)

    names = df.columns

    # ADD 'warming_level' threshold if absent. For now assume scenarios like ssp1_2p0 ==> warming level = 2.0 
    # ...also replace scenario with the ssp scenario only
    if "warming_level" not in names:
        assert 'scenario' in names, "Input table must contain `warming_level` or a `scenario` column of the form `ssp1_2p0`"        
        df['warming_level'], df['scenario'] = _parse_warming_level_and_ssp(df["scenario"].values)

    # # Also add missing fields that are not actually mandatory but expected in various subfunctions
    # for field in ["variable", "region", "scenario", "model"]:
    #     if field not in df:
    #         df[field] = ""

    return df


def _get_impact_parser():

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Impact indicator')
    group.add_argument("-v", "--variable", nargs="*")
    group.add_argument("--region")
    # group.add_argument("--format", default="ixmp4", choices=["ixmp4", "cie"])
    group.add_argument("--impact-file", nargs='+', default=[], 
        help=f'Files such as produced by Werning et al 2014 (.csv with ixmp4 standard). Also accepted is a glob * pattern to match downloaded datasets (see also rime-download-ls).')
    group.add_argument("--impact-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --impact-filter scenario='ssp2*'")
    group.add_argument("--model", nargs="+", help="if provided, only consider a set of specified model(s)")
    group.add_argument("--experiment", nargs="+", help="if provided, only consider a set of specified experiment(s)")

    group = parser.add_argument_group('Impact indicator (CIE)')
    group.add_argument("--subregion", help="if not provided, will default to region average")
    group.add_argument("--list-subregions", action='store_true', help="print all subregions and exit")
    group.add_argument("--weights", default='LonLatWeight', choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--season", default='annual', choices=list(CONFIG["preprocessing.seasons"]))

    return parser


def _get_impact_data(o, parser):

    if not o.impact_file:
        parser.error("the following argument is required: --impact-file")
        parser.exit(1)

    # Load impact data
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
    filter_kw = {} 
    for k, v in o.impact_filter:
        filter_kw.setdefault(k, [])
        filter_kw[k].append(v)

    if o.variable:
        filter_kw["variable"] = o.variable
    if o.region:
        filter_kw["region"] = o.region

    if o.pyam:
        import pyam
        impact_data_table = pyam.concat([pyam.IamDataFrame(f).filter(**filter_kw) for f in filtered_files])
    else:
        impact_data_table = concat([FastIamDataFrame.load(f).filter(**filter_kw) for f in filtered_files])

    if len(impact_data_table.variable) == 0:
        logger.error(f"Empty climate impact file with variable: {o.variable} and region {o.region}")
        parser.exit(1)


    # # Only handle one variable at a time?
    # if not o.impact_resample and len(impact_data_table.variable) > 1:
    #     print(f"More than one variable found.\n {sep.join(impact_data_table.variable)}\nPlease restrict the --variable filter.")
    #     parser.exit(1)


    # # Only handle one region at a time?
    # if len(impact_data_table.region) > 1:
    #     print(f"More than one region found.\n {sep.join(impact_data_table.region)}\nPlease restrict the --region filter.")
    #     parser.exit(1)

    # Convert to DataFrame
    impact_data_frame = impact_data_table.as_pandas()
    impact_data_frame = homogenize_table_names(impact_data_frame)

    return impact_data_frame    


def main():

    gmt_parser = _get_gmt_parser(gmt_ensemble=True)
    impact_parser = _get_impact_parser()

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, gmt_parser, impact_parser])
    
    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--running-mean-window", default=CONFIG["preprocessing.running_mean_window"])
    # group.add_argument("--warming-level-file", default=None)

    group = parser.add_argument_group('Aggregation')
    group.add_argument("--average-scenarios", action="store_true")
    group.add_argument("--equiprobable-models", action="store_true", help="if True, each model will have the same probability")
    group.add_argument("--quantiles", nargs='+', default=CONFIG["emulator.quantiles"], help="(default: %(default)s)")
    group.add_argument("--match-year-population", action="store_true")
    group.add_argument("--warming-level-step", default=CONFIG.get("preprocessing.warming_level_step"), type=float,
        help="Impact indicators will be interpolated to match this warming level (default: %(default)s)")
    group.add_argument("--impact-resample", action="store_true", 
        help="""Fit a distribution to the impact data from which to resample. 
        Assumes the quantile variables are named "{NAME}|5th percentile" and "{NAME}|95th percentile".""")
    group.add_argument("--impact-dist", default="auto", 
        choices=["auto", "norm", "lognorm"], 
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--impact-samples", default=100, type=int, help="Number of samples to draw if --impact-fit is set")

    group = parser.add_argument_group('Result')
    group.add_argument("-O", "--overwrite", action='store_true', help='overwrite final results')
    # group.add_argument("--backend-isimip-bins", nargs="+", default=CONFIG["preprocessing.isimip_binned_backend"], choices=["csv", "feather"])
    # parser.add_argument("--overwrite-isimip-bins", action='store_true', help='overwrite the intermediate calculations (binned isimip)')
    # parser.add_argument("--overwrite-all", action='store_true', help='overwrite intermediate and final')
    group.add_argument("-o", "--output-file", required=True)
    group.add_argument("--save-impact-table", help='file name to save the processed impacts table (e.g. for debugging)')

    parser.add_argument("--pyam", action="store_true", help='use pyam instead of own wrapper')

    o = parser.parse_args()

    setup_logger(o)

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exist. Use -O or --overwrite to reprocess.")
        parser.exit(0)

    # Load GMT data
    gmt_ensemble = _get_gmt_ensemble(o, parser)
    impact_data_frame = _get_impact_data(o, parser)

    # Now convert into a list of records
    impact_data_records = impact_data_frame.to_dict('records')

    if o.average_scenarios:
        from rimeX.preproc.digitize import average_per_group
        logger.info("average across scenarios (and years)...")
        impact_data_records = average_per_group(impact_data_records, by=("variable", "region", 'model', 'warming_level', 'year'))
        logger.info("average across scenarios (and years)...done")

    # Harmonize weights
    if o.equiprobable_models:
        logger.info("Normalization to give equal weight for each model per temperature bin...")        
        make_equiprobable_groups(impact_data_records, by=["variable", "region", "model", "warming_level"])
        logger.info("Normalization to give equal weight for each model per temperature bin...done")        

    # Interpolate records
    if o.warming_level_step:
        logger.info("Impact data: interpolate warming levels...")
        impact_data_records = interpolate_warming_levels(impact_data_records, o.warming_level_step,
            by = ["variable", "region", "scenario", "year", "model"])
        logger.info("Impact data: interpolate warming levels...done")

    # For population dataset the year can be matched to temperatrure time-series. It must be interpolated to yearly values first.
    if o.match_year_population:
        logger.info("Impact data: interpolate years...")
        impact_data_records = interpolate_years(impact_data_records, gmt_ensemble.index, 
            by=['variable', "region", 'warming_level', 'scenario', 'model'])
        logger.info("Impact data: interpolate years...done")

    # Fit and resample impact data if required
    if o.impact_resample:
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...")
        try:
            impact_data_records = fit_records(impact_data_records, o.impact_samples, o.impact_dist, 
                by=["variable", "region", "warming_level", "year", "scenario", "model"])
        except Exception as error:
            parser.exit(1)
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...done")

    if o.save_impact_table:
        logger.info("Save impact table...")
        pd.DataFrame(impact_data_records).to_csv(o.save_impact_table, index=None)
        logger.info("Save impact table...done")


    # Only use future values to avoid getting in trouble with the warming levels.
    gmt_ensemble = gmt_ensemble.loc[2015:]  

    assert np.isfinite(gmt_ensemble.values).all(), 'some NaN in MAGICC run'

    # Recombine GMT ensemble with binned ISIMIP data
    quantiles = recombine_gmt_ensemble(impact_data_records, gmt_ensemble, o.quantiles, match_year=o.match_year_population)

    # GMT result to disk
    logger.info(f"Write output to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    quantiles.to_csv(o.output_file)


if __name__ == "__main__":
    main()