"""
For a given scenario, return the mapped percentile for an indicator
"""
import tqdm
from itertools import groupby
import numpy as np
import pandas as pd
import xarray as xa

from rimeX.logs import logger
from rimeX.compat import get_rename_mapping, _get_ssp_mapping

# from rimeX.compat import (FastIamDataFrame, concat, read_table, _isnumerical, _simplify, homogenize_table_names)
# from rimeX.datasets import get_datapath


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
            logger.debug("ImpactDataInterpolator: convert Dataset to DataArray")
            dataarray = dataarray.to_array("variable")

        logger.debug(f"ImpactDataInterpolator: input dimensions of DataArray: {dataarray.dims}")
        logger.debug(f"ImpactDataInterpolator: input coordinates of DataArray: {list(dataarray.coords)}")

        mapping = get_rename_mapping(dataarray.dims)
        logger.debug(f"ImpactDataInterpolator: rename: {mapping}")
        dataarray = dataarray.rename(mapping)

        assert "warming_level" in dataarray.dims

        if "ssp_family" in dataarray.dims: 
            logger.debug("ImpactDataInterpolator: ssp_family found in DataArray dims")
            dataarray = dataarray.assign_coords({"ssp_family": _get_ssp_mapping(dataarray.ssp_family.values)})
        elif "scenario" in dataarray.dims:
            logger.debug("ImpactDataInterpolator: ssp_family derived from scenario dim")
            dataarray = dataarray.assign_coords({"ssp_family": xa.DataArray(_get_ssp_mapping(dataarray.scenario.values), dims=["scenario"])})
        else:
            logger.debug(f"ImpactDataInterpolator: ssp_family not found in input DataArray: {dataarray.dims}")

        indices = [c for c in ["warming_level", "year", "ssp_family"] if c in dataarray.dims]
        logger.debug(f"ImpactDataInterpolator: transpose dataarray {indices}...")
        dataarray = dataarray.transpose(*indices, ...)

        self.dataarray = dataarray
        self.kwargs = kwargs


    @classmethod
    def from_dataframe(cls, table, mapping=None, meta_levels=None, index_levels=None, **kwargs):

        if mapping:
            table = table.rename(mapping or {}, axis=1)

        logger.debug(f"ImpactDataInterpolator.from_dataframe: table columns: {list(table.columns)}")            

        if index_levels is None:
            index_levels = [c for c in ['warming_level', 'year'] if c in table.columns]

        if meta_levels is None:
            meta_levels = [c for c in ['region', 'model', 'variable', 'scenario'] if c in table.columns and c not in index_levels]

        if "warming_level" not in table.columns:
            raise ValueError("impact table must contain `warming_level`")

        # logger.debug(f"ImpactDataInterpolator.from_dataframe: {table}")

        # Create a 2-D data frame indexed by year and warming level
        # (this is usually very fast)        
        logger.debug("ImpactDataInterpolator.from_dataframe: reshape impact table with multi indices")
        levels = index_levels + meta_levels
        series = table.set_index(levels)['value'];

        if not series.index.is_unique:
            logger.warning("ImpactDataInterpolator.from_dataframe: index is not unique: drop duplicates")
            series = table.drop_duplicates(levels).set_index(levels)['value']

        # logger.debug(f"ImpactDataInterpolator.from_dataframe: series: {series}")
        logger.debug("ImpactDataInterpolator.from_dataframe: transform to xarray.DataArray")
        dataarray = xa.DataArray.from_series(series)

        return cls(dataarray, **kwargs)


    def hasyear(self):
        return "year" in self.dataarray.dims

    def __call__(self, values, **kwargs):
        return self.interpolate_scipy(values, **{**self.kwargs, **kwargs})


    def interpolate_scipy(self, gmt_table, ignore_ssp=False, **kwargs):

        if not ignore_ssp:
            if "ssp_family" not in self.dataarray.dims:
                ignore_ssp = True
                logger.debug("No SSP information found in impact data: ignore")

        if not ignore_ssp:
            if "ssp_family" in gmt_table:
                gsat_ssp_family = _get_ssp_mapping(gmt_table["ssp_family"].values)
            elif "scenario" in gmt_table:
                gsat_ssp_family = _get_ssp_mapping(gmt_table["scenario"].values)
            else:
                ignore_ssp = True
                logger.debug("No SSP information found in GSAT data: ignore")

        if ignore_ssp:
            return self._interpolate_scipy(gmt_table, **kwargs)

        # Group by ssp family and combine
        ssp_family = self.dataarray.ssp_family.values

        diff = set(ssp_family).symmetric_difference(gsat_ssp_family)
        if diff:
            logger.warning(f"SSP family present in only GSAT or only impact data: {diff}. It will be ignored (see also --ignore-ssp)")

        return_dataarray = kwargs.pop("return_dataarray", False)

        data = []
        for ssp in set(ssp_family).intersection(gsat_ssp_family):
            logger.info(f"recombine_gmt_table for {ssp}")
            interp_ssp = ImpactDataInterpolator(self.dataarray.sel(ssp_family=[ssp]), **self.kwargs)
            gsat_ssp = gmt_table.iloc[gsat_ssp_family == ssp][[c for c in gmt_table if c != "ssp_family"]]
            # gsat_ssp = gmt_table.iloc[gsat_ssp_family == ssp]
            res = interp_ssp._interpolate_scipy(gsat_ssp, return_dataarray=True, **kwargs)
            # res['ssp_family'] = ssp
            data.append( res.to_series() )

        data = pd.concat(data)

        if not return_dataarray:
            return data.reset_index(name='value')

        else:
            return xa.DataArray.from_series(data)


    def _interpolate_scipy(self, gmt_table, method="linear", mapping=None, return_dataarray=False, ignore_year=False, bounds_error=False):

        from scipy.interpolate import RegularGridInterpolator

        logger.debug(f"input gmt_table columns: {gmt_table.columns}")
        logger.debug("rename gmt_table columns")
        gmt_table = gmt_table.rename({"value":"warming_level", **(mapping or {})}, axis=1)

        gmt = gmt_table['warming_level'].values

        index_levels = ["warming_level"]
        meta_levels = [c for c in self.dataarray.dims if c not in ["warming_level", "year", "ssp_family"]]

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

        logger.debug(f"index levels for interp: {index_levels}")
        logger.debug(f"meta levels for interp: {meta_levels}")

        logger.debug("build indices")

        index = []

        logger.debug("build warming_level index")
        index.append(gmt)

        if "year" in index_levels:
            gmt_year = gmt_table['year'].values
            index.append(gmt_year)

        indices = np.array(index).T

        interp = RegularGridInterpolator([self.dataarray.coords[k].values for k in index_levels], self.dataarray.transpose(*index_levels, ...).values, bounds_error=bounds_error, method=method)
        values = interp(indices, method=method)

        # first step build a self.dataarray
        logger.debug("rebuild a DataArray")
        other_dims = [d for d in self.dataarray.dims if d not in index_levels]
        all_other_dims = [d for d in self.dataarray.coords if d not in index_levels]
        logger.debug(f"dimensions inherited from impact table {other_dims}")
        logger.debug(f"coordinates inherited from impact table {all_other_dims}")
        data = xa.DataArray(values, dims=['index']+other_dims, coords={k:v for k, v in self.dataarray.coords.items() if k in all_other_dims})

        # Provide the detail of the multi-index
        gmt_index_names = [c for c in ["year", "ssp_family", "warming_level", "model", "scenario", "quantile", "percentile"] if c in gmt_table.columns]

        # ...also add other info like model and scenario, but rename them to avoid any conflict with the impact table
        gmt_index_rename = {"model": "gsat_model", "scenario": "gsat_scenario", "ssp_family": "gsat_ssp_family", "quantile": "gsat_quantile", "percentile": "gsat_percentile"}
        gmt_index_names = [gmt_index_rename.get(c, c) for c in gmt_index_names]
        logger.debug(f"dimensions inherited from gsat table {gmt_index_names}")

        midx = xa.Coordinates.from_pandas_multiindex(
            gmt_table.rename(gmt_index_rename, axis=1).set_index(gmt_index_names).index, 'index')
        data = data.assign_coords(midx)
        
        # logger.debug(f"full dataarray {data}")

        if return_dataarray:
            return data

        # ... now flatten to a Dataframe
        logger.debug("transform to DataFrame")
        df = data.to_series().reset_index(name='value')
        return df


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