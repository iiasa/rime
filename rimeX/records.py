"""This module defines functions that operate on a list of records
"""

from pathlib import Path
import datetime
import argparse
import glob
import tqdm
import itertools
from itertools import groupby, product
import numpy as np
import pandas as pd
# import xarray as xa

from rimeX.logs import logger, setup_logger, log_parser
from rimeX.config import CONFIG, config_parser


def interpolate_warming_levels(impact_data_records, warming_level_step, by):
    key_fn = lambda r: tuple(r.get(k, 0) for k in by)
    interpolated_records = []
    for key, group in groupby(sorted(impact_data_records, key=key_fn), key=key_fn):
        igwls, ivalues = np.array([(r['warming_level'], r['value']) for r in sorted(group, key=lambda r: r['warming_level'])]).T
        gwls = np.arange(min(igwls), max(igwls)+warming_level_step, warming_level_step)
        values = np.interp(gwls, igwls, ivalues)
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
            meta = {k:','.join(sorted(set(str(r.get(k)) for r in group))) for k in first_record.keys() if k not in ["value", "midyear", "weights"]}
        else:
            meta = {k:v for k,v in zip(by, key)}
        years, values, weights = np.array([[r["midyear"], r["value"], r.get("weight", 1)] for r in group]).T
        total_weight = weights.sum()
        mean_value = (values*weights).sum()/total_weight
        mean_year = (years*weights).sum()/total_weight
        mean_weight = (weights*weights).sum()/total_weight  

        average_records.append({"value": mean_value, "midyear": mean_year, "weight": mean_weight, **meta})

    return average_records


def drop_dims(records, dims):
    """in-place dropping of dimensions
    """
    for r in records:
        for dim in dims:
            r.pop(dim)

def pool_dims(records, by, newdim, sep="|", integer_value=False, keep=False):
    """Pool dimensions

    Parameters
    ----------
    records: list of dict with keys "year", "value" [, "weight"] and elements of `by`
    dims: list of keys for grouping
    newdim: str, optional
        pooled dimension name
    sep: str, "|" by default
        separator to create new dim's value
    integer_value: bool, False by default
        if True, simply assign an integer value to the new dimension (optimization)
    keep: bool, optional
        keep old dimensions

    Returns
    -------
    pooled_records: (weighted) list of records
    """
    pooled_records = []
    key_pool = lambda r: tuple(r.get(k, 0) for k in by)
    for i, (key, group) in enumerate(groupby(sorted(records, key=key_pool), key=key_pool)):
        newvalue = i if integer_value else sep.join(str(val) for val in key)
        factor = {k:v for k,v in r.items() if keep or k not in by}
        for r in group:
            pooled_records.append({newdim: newvalue, **factor})

    return pooled_records


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


def make_models_equiprobable(records):
    make_equiprobable_groups(records, by=[ "model", "variable", "region", "warming_level" ])


def filter_first_value(records, keys=["model", "experiment", "warming_level", "ensemble", "variable", "season", "region", "subregion"]):
    """Filter records to keep only the first value for each warming level
    """
    key_fn = lambda r: tuple(r.get(k, 0) for k in keys)
    return [next(group) for key, group in groupby(sorted(records, key=key_fn), key=key_fn)]


def add_peaking_tag(warming_level_records,
                       keys=["model", "experiment", "warming_level", "ensemble"]):
    """Filter records to keep only the value before peaking, if any
    """
    # first check if there is a peak in the input experiment
    key_fn = lambda r: tuple(r.get(k, 0) for k in keys)

    for _, group in groupby(sorted(warming_level_records, key=key_fn), key=key_fn):
        timeseries = list(group)
        years, values = np.array([(r['year'], r['actual_warming']) for r in timeseries]).T
        assert (years[1:] > years[:-1]).all(), "years are not sorted"
        i = np.argmax(values)
        for j, r in enumerate(timeseries):
            r['post-peaking'] = j > i


def filter_peaking(records, warming_level_records,
                       keys=["model", "experiment", "warming_level", "ensemble"]):
    """Split records into pre-peaking and post-peaking records

    Returns
    -------
    records_pre, records_post
    """
    if "post-peaking" not in warming_level_records[0]:
        add_peaking_tag(warming_level_records, keys=keys)

    key_fn = lambda r: tuple(r.get(k, 0) for k in keys)

    pre_keys = set(key_fn(r) for r in warming_level_records if not r['post-peaking'])
    post_keys = set(key_fn(r) for r in warming_level_records if r['post-peaking'])

    return [r for r in records if key_fn(r) in pre_keys], [r for r in records if key_fn(r) in post_keys]
