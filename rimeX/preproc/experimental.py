"""Experimental code not ready for production, which may become deprecated.
"""
from itertools import groupby
import math
import numpy as np
from scipy.stats import norm

from rimeX.logs import logger


def calculate_interannual_variability_standard_deviation(all_annual, running_mean_window, start_year=None):
    """
    Parameters
    ----------
    all_annual: DataFrame with all experiments but historical runs
    running_mean_window: 21 or 31 years typically (to define "climate")
    start_year: set in config.toml as "temperature_sigma_first_year", i.e. 2015 for CMIP6
        => used to exclude historical values from the calculation

    Returns
    -------
    sigma: float, standard deviation

    Notes
    -----
    by default exclude historical values from the calculation of the standard deviation.
    That's because 1) historical runs have slightly different forcings (volcanic activity) 
    and 2) we'll use that threshold on projections mainly
    """
    if start_year is not None:
        all_annual = all_annual.loc[start_year:]

    residuals = []
    for experiment in all_annual:
        annual = all_annual[experiment].dropna()
        smoothed = annual.rolling(running_mean_window, center=True).mean()
        residuals.append((annual - smoothed).dropna().values)
    return np.concatenate(residuals).std()


def get_matching_years_by_temperature_bucket(model, all_annual, warming_levels, running_mean_window, 
    temperature_sigma_range, temperature_sigma_first_year, temperature_bin_size, projection_baseline):
    """
    """
    # Calculate model-specific interannual variability's standard deviation over all experiments including projections
    sigma = calculate_interannual_variability_standard_deviation(all_annual, running_mean_window, start_year=temperature_sigma_first_year)

    logger.info(f"{model}'s interannual variability S.D is {sigma:.2f} degrees C (to be multiplied by {temperature_sigma_range} on each side)")
    
    records = []

    all_smoothed = all_annual.rolling(running_mean_window, center=True).mean()    

    for experiment in all_annual:
        data = all_annual[experiment].dropna()
        y1, y2 = projection_baseline
        accumulated_warming = data.cumsum()
        warming_rate = all_smoothed[experiment].diff()
        accumulated_warming -= accumulated_warming.loc[y1:y2].mean()

        for wl in warming_levels:
            lo = wl - temperature_sigma_range * sigma - temperature_bin_size/2
            hi = wl + temperature_sigma_range * sigma + temperature_bin_size/2
            bucket = (data.values >= lo) & (data.values <= hi)
            if bucket.sum() > 0:
                logger.info(f"{model} | {experiment} | {wl} degrees : {bucket.sum()} years selected")
            for year, value, acc, rate in zip(data.index.values[bucket], data.values[bucket], accumulated_warming.values[bucket], warming_rate.values[bucket]):
                records.append({"model": model, "experiment": experiment, "warming_level": wl, "year": year, 
                    "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})

    return records


def get_matching_years_by_pure_temperature(model, all_annual, warming_levels, projection_baseline):
    """ 
    """    
    records = []

    all_smoothed = all_annual.rolling(21, center=True).mean()    

    for experiment in all_annual:
        data = all_annual[experiment].dropna()
        y1, y2 = projection_baseline
        accumulated_warming = data.cumsum()
        accumulated_warming -= accumulated_warming.loc[y1:y2].mean()
        # warming_rate = data.diff()
        warming_rate = all_smoothed[experiment].diff()        


        half_widths = np.diff(warming_levels)/2
        half_width = half_widths[0]
        bins = np.concatenate([warming_levels - half_width, [warming_levels[-1] + half_width]])
        indices = np.digitize(data.values, bins=bins)
        bad = (indices == 0) | (indices == bins.size)
        # indices = indices.clip(1, bins.size-1)
        for idx, year, value, acc, rate in zip(indices[~bad], data.index.values[~bad], 
            data.values[~bad], accumulated_warming.values[~bad], warming_rate.values[~bad]):
            wl = warming_levels[idx-1]
            records.append({"model": model, "experiment": experiment, "warming_level": wl, "year": year, 
                "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})

    return records


def resample_with_natural_variability(records,
    binsize,
    sigma=0.14,
    ):
    """Enlarge the warming_level: value mapping to account for natural variability

    Parameters
    ----------
    records: list of dict with at least the "warming_level" field
    binsize: global warming level binning, used together with sigma to determine cut-off
    sigma: standard deviation of interannual variability in GMT (default to 0.14)

    Returns
    -------
    extended_records: extended list of dict with "warming_level", "weight" and other fields 

    Note this function resets the `weight` attribute.
    """

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
