"""
"""
import os
import argparse
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from rimeX.preproc.regional_average import get_files, isimip_parser
from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser

def global_mean_file(variable, model, experiment, root=None):
    if root is None: root = CONFIG["isimip.climate_impact_explorer"]
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


def get_matching_years_by_time_bucket(model, all_annual, warming_levels, running_mean_window, projection_baseline):

    all_smoothed = all_annual.rolling(running_mean_window, center=True).mean()
    records = []

    for experiment in all_annual:

        smoothed = all_smoothed[experiment].dropna()
        warming_rate = all_smoothed[experiment].diff()
        accumulated_warming = all_annual[experiment].cumsum()
        y1, y2 = projection_baseline
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

            # also calculate mean warming during that period, for diagnostic purposes
            value = all_annual[experiment].reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
            acc = accumulated_warming.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
            rate = warming_rate.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
            records.append({"model": model, "experiment": experiment, "warming_level":  wl, 
                "year": smoothed.index[idx], "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})

    return records


def get_warming_level_file(running_mean_window, warming_level_name=None, **kw):
    if warming_level_name is None:
        warming_level_name = f"warming_level_running-{running_mean_window}-years.csv"
    # return Path(__file__).parent / warming_level_folder / warming_level_name
    return Path(CONFIG["isimip.climate_impact_explorer"]) / "warming_levels" / warming_level_name

def main():
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, isimip_parser])
    
    egroup = parser.add_mutually_exclusive_group()
    group = egroup.add_argument_group('warming levels')
    group.add_argument("--min-warming-level", type=float, default=CONFIG["emulator.warming_level_min"])
    group.add_argument("--step-warming-level", type=float, default=CONFIG["emulator.warming_level_step"])
    group.add_argument("--max-warming-level", type=float, default=CONFIG.get("emulator.warming_level_max", 10))
    egroup.add_argument("--warming-levels", nargs='*', type=float)

    parser.add_argument("--running-mean-window", type=int, default=CONFIG["emulator.running_mean_window"])
    parser.add_argument("--projection-baseline", nargs=2, type=int, default=CONFIG["emulator.projection_baseline"])
    parser.add_argument("--projection-baseline-offset", type=float, default=CONFIG["emulator.projection_baseline_offset"])

    parser.add_argument("-o", "--output-file")
    parser.add_argument("-O", "--overwrite", action="store_true")

    # parser.add_argument("-o", "--output-file", default=CONFIG["emulator.warming_level_file"])

    o = parser.parse_args()
    setup_logger(o)

    if o.output_file is None:
        o.output_file = get_warming_level_file(**{**CONFIG, **vars(o)})

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exists. Use -O or --overwrite to reprocess.")
        return

    if o.warming_levels is not None:
        warming_levels = o.warming_levels
    else:
        warming_levels = np.arange(o.min_warming_level, o.max_warming_level, o.step_warming_level).round(2)

    records = []

    for model in o.model:

        all_annual = load_annual_values(model, o.experiment, projection_baseline=o.projection_baseline, projection_baseline_offset=o.projection_baseline_offset)
        records.extend(get_matching_years_by_time_bucket(model, all_annual, warming_levels, o.running_mean_window, o.projection_baseline))

    df = pd.DataFrame(records)

    logger.info(f"Write to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(o.output_file, index=None)

if __name__ == "__main__":
    main()