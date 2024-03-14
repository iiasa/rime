"""
"""
import os
import argparse
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from regional_average import get_files
from logs import logger, log_parser
from config import config, config_parser

def global_mean_file(variable, model, experiment, root=config["climate_impact_explorer"]):
    return Path(root) / f"isimip_global_mean/{variable}/globalmean_{variable}_{model.lower()}_{experiment}.csv"


def load_annual_values(model, experiments, variable="tas", projection_baseline=None, projection_baseline_offset=None):
    """Load all experiments prepended with historical values
    """
    read_annual_values_experiment = lambda experiment: pd.read_csv(global_mean_file("tas", model, experiment), index_col=0)[variable].rolling(12).mean()[11::12]

    historical = read_annual_values_experiment("historical")

    # Load all experiments to calculate things like natural variability (temperature method)
    # prepend historical values because of the 
    all_annual = {}

    for experiment in experiments:
        if experiment == "historical": 
            continue
        try:
            all_annual[experiment] = pd.concat([historical, read_annual_values_experiment(experiment)])
        except FileNotFoundError:
            logger.warning(f"{model} | {experiment} ==> not found")
            continue

    df = pd.DataFrame(all_annual)
    df.index = [datetime.datetime.fromisoformat(dt).year for dt in df.index]

    # Align historical temperature objections with projection baseline
    if projection_baseline is not None:
        y1, y2 = projection_baseline
        df -= df.loc[y1:y2].mean()

        if projection_baseline_offset is not None:
            df += projection_baseline_offset

    return df


def get_matching_years_by_time_bucket(model, all_annual, warming_levels, running_mean_window, all_years=True):

    all_smoothed = all_annual.rolling(running_mean_window, center=True).mean()
    records = []

    for experiment in all_annual:

        smoothed = all_smoothed[experiment].dropna()
        warming_rate = all_smoothed[experiment].diff()
        accumulated_warming = all_annual[experiment].cumsum()
        y1, y2 = config['projection_baseline']
        accumulated_warming -= accumulated_warming.loc[y1:y2].mean()

        sort = np.argsort(smoothed.values)
        rank = np.searchsorted(smoothed.values, warming_levels, sorter=sort)
        bad = rank == smoothed.size
        matching_years_idx = sort[rank[~bad]]

        # matching_years_idx = np.searchsorted(smoothed, warming_levels)
        for idx, wl in zip(matching_years_idx, warming_levels[~bad]):
            # never reached the warming level, do not record
            found = smoothed.values[idx]
            if found < wl:
                logger.warning(f"{model} | {experiment} never reached {wl} degrees")
                break

            # Write down all individual years
            if all_years:
                central_year = smoothed.index[idx]
                for y in range(central_year-running_mean_window//2, central_year+running_mean_window//2+1):
                    value = all_annual[experiment].loc[y]
                    acc = accumulated_warming.loc[y]
                    rate = warming_rate.loc[y]
                    records.append({"model": model, "experiment": experiment, "warming_level":  wl, 
                        "year": y, "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})
            else:
                # also calculate mean warming during that period, for diagnostic purposes
                value = all_annual[experiment].reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                acc = accumulated_warming.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                rate = warming_rate.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                records.append({"model": model, "experiment": experiment, "warming_level":  wl, 
                    "year": smoothed.index[idx], "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})

    return records


def calculate_interannual_variability_standard_deviation(all_annual, running_mean_window, start_year=None):
    """
    Parameters
    ----------
    all_annual: DataFrame with all experiments but historical runs
    running_mean_window: 21 or 31 years typically (to define "climate")
    start_year: set in config.yml as "temperature_sigma_first_year", i.e. 2015 for CMIP6
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
    temperature_sigma_range, temperature_sigma_first_year, temperature_bin_size):
    """
    """
    # Calculate model-specific interannual variability's standard deviation over all experiments including projections
    sigma = calculate_interannual_variability_standard_deviation(all_annual, running_mean_window, start_year=temperature_sigma_first_year)

    logger.info(f"{model}'s interannual variability S.D is {sigma:.2f} degrees C (to be multiplied by {temperature_sigma_range} on each side)")
    
    records = []

    all_smoothed = all_annual.rolling(running_mean_window, center=True).mean()    

    for experiment in all_annual:
        data = all_annual[experiment].dropna()
        y1, y2 = config['projection_baseline']
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


def get_matching_years_by_pure_temperature(model, all_annual, warming_levels):
    """ 
    """    
    records = []

    all_smoothed = all_annual.rolling(21, center=True).mean()    

    for experiment in all_annual:
        data = all_annual[experiment].dropna()
        y1, y2 = config['projection_baseline']
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


def get_warming_level_file(matching_method, running_mean_window, temperature_sigma_range, warming_level_name=None, **kw):
    if warming_level_name is None:
        matching_method_label = f"{matching_method}-{temperature_sigma_range}s" if matching_method == "temperature" else matching_method
        natvarlab = "" if matching_method == "pure" else f"-bucket_climate-{running_mean_window}-years"
        warming_level_name = f"warming_level_year_by_{matching_method_label}{natvarlab}.csv"
    # return Path(__file__).parent / warming_level_folder / warming_level_name
    return Path(config["climate_impact_explorer"]) / "warming_levels" / warming_level_name

def main():
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser])
    # parser.add_argument("--variable", nargs="+", default=["tas"], choices=["tas"])
    parser.add_argument("--models", nargs="+", default=config["models"], choices=config["models"])
    parser.add_argument("--experiments", nargs="+", default=config["experiments"], choices=config["experiments"])
    
    egroup = parser.add_mutually_exclusive_group()
    group = egroup.add_argument_group('warming levels')
    group.add_argument("--min-warming-level", type=float, default=config["warming_level_min"])
    group.add_argument("--step-warming-level", type=float, default=config["warming_level_step"])
    group.add_argument("--max-warming-level", type=float, default=config.get("warming_level_max", 10))
    egroup.add_argument("--warming-levels", nargs='*', type=float)

    parser.add_argument("--running-mean-window", type=int, default=config['running_mean_window'])
    parser.add_argument("--projection-baseline", nargs=2, type=int, default=config['projection_baseline'])
    parser.add_argument("--projection-baseline-offset", type=float, default=config['projection_baseline_offset'])
    parser.add_argument("--matching-method", default=config["matching_method"], choices=["time", "temperature", "pure"])
    parser.add_argument("--temperature-sigma-range", type=float, default=config["temperature_sigma_range"])
    parser.add_argument("--temperature-sigma-first-year", type=int, default=config["temperature_sigma_first_year"])
    parser.add_argument("--temperature-bin-size", type=float, default=config["temperature_bin_size"])

    parser.add_argument("-o", "--output-file")
    parser.add_argument("-O", "--overwrite", action="store_true")

    # parser.add_argument("-o", "--output-file", default=config["warming_level_file"])

    o = parser.parse_args()

    if o.output_file is None:
        o.output_file = get_warming_level_file(**{**config, **vars(o)})

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exists. Use -O or --overwrite to reprocess.")
        return

    if o.warming_levels is not None:
        warming_levels = o.warming_levels
    else:
        warming_levels = np.arange(o.min_warming_level, o.max_warming_level, o.step_warming_level).round(2)

    records = []

    for model in o.models:

        all_annual = load_annual_values(model, o.experiments, projection_baseline=o.projection_baseline, projection_baseline_offset=o.projection_baseline_offset)

        if o.matching_method == "time":
            records.extend(get_matching_years_by_time_bucket(model, all_annual, warming_levels, o.running_mean_window))

        elif o.matching_method == "temperature":
            records.extend(get_matching_years_by_temperature_bucket(model, all_annual, warming_levels, o.running_mean_window, 
                o.temperature_sigma_range, o.temperature_sigma_first_year, o.temperature_bin_size))

        elif o.matching_method == "pure":
            records.extend(get_matching_years_by_pure_temperature(model, all_annual, warming_levels))

        else:
            raise NotImplementedError(o.matching_method)

    df = pd.DataFrame(records)

    logger.info(f"Write to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(o.output_file, index=None)

if __name__ == "__main__":
    main()