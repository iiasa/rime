"""
"""
import os
import argparse
import json
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from rimeX.preproc.regional_average import get_files, isimip_parser
from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser
from rimeX.preproc.global_average_tas import global_mean_file

def load_annual_values(model, experiments, variable="tas", projection_baseline=None, projection_baseline_offset=None, simulation_round=None):
    """Load all experiments prepended with historical values
    """
    read_annual_values_experiment = lambda experiment: pd.read_csv(global_mean_file("tas", model, experiment, simulation_round), index_col=0)[variable].rolling(12).mean()[11::12]

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

    if not all_annual:
        raise ValueError(f"No experiments found for {model} :: {variable} ({experiments})")

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
    """

    Next to the warming levels and year, the output contains the actual warming, rate of warming and accumulated warming.
    Note that while the accumulated warming is calculated on the temperature above pre-industrial,
    it is zeroed at the projection baseline, mostly for practical purpose, because we only calculated the values since 1980s.
    (similarly the GMT above PI is also aligned to the projection baseline, using observed warming during the baseline period)
    The advantage of this approach is that the models align with each other for the present-days. The trade-off is that we
    "erase" the memory since 1850, and any time-lagged effect of the warming inside these models.
    """

    all_smoothed = all_annual.rolling(running_mean_window, center=True).mean()
    records = []

    for experiment in all_annual:

        smoothed = all_smoothed[experiment].dropna()
        warming_rate = all_smoothed[experiment].diff()

        accumulated_warming = all_annual[experiment].cumsum()
        y1, y2 = projection_baseline
        accumulated_warming -= accumulated_warming.loc[y1:y2].mean()


        # matching_years_idx = np.searchsorted(smoothed, warming_levels)
        for wl in warming_levels:
            step = warming_levels[1] - warming_levels[0]
            match = (wl - step / 2 <= smoothed.values) & (smoothed.values < wl + step/2)

            # never reached the warming level, do not record
            if not match.any():
                logger.warning(f"{model} | {experiment} never reached {wl} degrees")
                break

            for idx in np.where(match)[0]:
                # also calculate mean warming during that period, for diagnostic purposes
                # value = all_annual[experiment].reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                value = smoothed.values[idx]
                acc = accumulated_warming.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                rate = warming_rate.reindex(smoothed.index).values[idx-running_mean_window//2:idx+running_mean_window//2+1].mean()
                records.append({"model": model, "experiment": experiment, "warming_level":  wl,
                    "year": smoothed.index[idx], "actual_warming": value, "accumulated_warming": acc, "warming_rate": rate})

    return records


def get_root_directory(running_mean_window=None, config_name=None, tag=None, simulation_round=None, **kw):
    if config_name is None: config_name = CONFIG.get("preprocessing.config_name")
    if config_name:
        return config_name
    if tag is None: tag = CONFIG.get("preprocessing.tag")
    if running_mean_window is None: running_mean_window = CONFIG["preprocessing.running_mean_window"]
    if simulation_round is None: simulation_round = CONFIG.get("isimip.simulation_round")
    simulation_round = "-".join([{"isimip2b": "isimip2", "isimip3b": "isimip3"}.get(s.lower(), s.lower()) for s in simulation_round])
    tags = [f"running-{running_mean_window}-years", tag]
    return Path(CONFIG["isimip.climate_impact_explorer"]) / simulation_round / "_".join(str(tag) for tag in tags if tag)

def get_warming_level_file(warming_level_path=None, **kw):
    if warming_level_path is None:
        root_directory = get_root_directory(**kw)
        warming_level_path = root_directory / "warming_levels.csv"
    # return Path(__file__).parent / warming_level_folder / warming_level_path
    return warming_level_path


def main():
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, isimip_parser])

    egroup = parser.add_mutually_exclusive_group()
    group = egroup.add_argument_group('warming levels')
    group.add_argument("--min-warming-level", type=float, default=CONFIG["preprocessing.warming_level_min"])
    group.add_argument("--step-warming-level", type=float, default=CONFIG["preprocessing.warming_level_step"])
    group.add_argument("--max-warming-level", type=float, default=CONFIG.get("preprocessing.warming_level_max", 10))
    egroup.add_argument("--warming-levels", nargs='*', type=float)

    parser.add_argument("--running-mean-window", type=int, default=CONFIG["preprocessing.running_mean_window"])

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

        try:
            all_annual = load_annual_values(model, o.experiment, projection_baseline=o.projection_baseline, projection_baseline_offset=o.projection_baseline_offset)
        except ValueError as e:
            logger.warning(e)
            logger.warning(f"Skip {model}")
            continue
        records.extend(get_matching_years_by_time_bucket(model, all_annual, warming_levels, o.running_mean_window, o.projection_baseline))

    df = pd.DataFrame(records)

    logger.info(f"Write to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(o.output_file, index=None)

    # Now save the config file for documentation and to ensure consistency with binned ISIMIP outputs
    json_file = str(o.output_file).replace(".csv", ".json")
    logger.info(f"Write config info to {json_file}")
    small_config ={k: v for k, v in CONFIG.items() if k.endswith((
        "running_mean_window",
        "projection_baseline",
        "projection_baseline_offset",
        ))}
    with open(json_file, "w") as f:
        json.dump(small_config, f, indent=2, sort_keys=True, default=str, ensure_ascii=False, separators=(',', ': '), )

if __name__ == "__main__":
    main()