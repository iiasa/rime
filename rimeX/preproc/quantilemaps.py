"""Quantile maps for impacts that do not stem from a regional average
"""

from itertools import groupby
import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xa

from rimeX.config import CONFIG, config_parser
from rimeX.logs import logger, log_parser
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.preproc.warminglevels import get_warming_level_file, get_root_directory
from rimeX.preproc.digitize import transform_indicator


def make_quantile_map_array(indicator:Indicator, warming_levels:pd.DataFrame,
                            quantile_bins=10, season="annual", running_mean_window=21,
                            projection_baseline=None):

    simulations = indicator.simulations
    w = running_mean_window // 2
    quant_bins = quantile_bins
    quant_step = 1/quant_bins
    quants = np.linspace(quant_step/2, 1-quant_step/2, quant_bins)

    warming_level_data = []
    warming_level_coords = []

    keywl = lambda r: r["warming_level"]
    wl_records = sorted(warming_levels.to_dict(orient="records"), key=keywl)

    logger.info(f"Process quantile maps for {indicator.name} | {season}. Warming levels {wl_records[0]['warming_level']} to {wl_records[-1]['warming_level']}")

    for wl, group in groupby(wl_records, key=keywl):

        logger.info(f"==== {wl} ====")
        files_to_concat = []
        group = list(group)

        for r in tqdm.tqdm(group):
            simus_historical = [s for s in simulations if _matches(s["climate_scenario"], "historical") and _matches(s["climate_forcing"], r["model"])]
            simus = [s for s in simulations if _matches(s["climate_scenario"], r["experiment"]) and _matches(s["climate_forcing"], r["model"])]
            if len(simus_historical) == 0 or len(simus) == 0:
                logger.warning(f"No simulation for {indicator.name} {r['model']} {r['experiment']}: Skip")
                continue
            assert len(simus_historical) == 1
            assert len(simus) == 1
            simu_historical = simus_historical[0]
            simu = simus[0]
            # warming_level	year
            filepath_hist = indicator.get_path(**simu_historical)
            filepath = indicator.get_path(**simu)

            with xa.open_mfdataset([filepath_hist, filepath], combine='nested', concat_dim="time") as ds:

                # only select relevant months
                if season is not None:
                    season_mask = ds["time.month"].isin(CONFIG["preprocessing.seasons"][season])
                else:
                    season_mask = slice(None)

                seasonal_sel = ds[indicator.name].isel(time=season_mask)

                # mean over the ref period
                if projection_baseline is not None:
                    y1, y2 = projection_baseline
                    dataref = seasonal_sel.sel(time=slice(str(y1),str(y2))).mean("time")

                # mean over required time-slice
                data = seasonal_sel.sel(time=slice(str(r['year']-w),str(r['year']+w))).mean("time")

                # subtract the reference period or express as relative change
                data = transform_indicator(data, indicator.name, dataref=dataref).load()

                # assign metadata
                data = data.assign_coords({
                    "warming_level": r["warming_level"],
                    "model": r["model"],
                    "experiment": r["experiment"],
                    "midyear": r["year"],
                    })

            files_to_concat.append(data)

        samples = xa.concat(files_to_concat, dim="sample")
        quantiles = samples.quantile(quants, dim="sample")
        warming_level_data.append(quantiles)
        warming_level_coords.append(wl)

    warming_level_data = xa.concat(warming_level_data, dim="warming_level")
    warming_level_data = warming_level_data.assign_coords({"warming_level": warming_level_coords}) # otherwise it's dropped apparently
    warming_level_data.name = indicator.name
    return warming_level_data


def get_filepath(name, season="annual", root_dir=None, **kw):
    if root_dir is None:
        root_dir = get_root_directory(**kw)
    return root_dir / "quantilemaps" / name / f"{name}_{season}_quantilemaps.nc"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[config_parser, log_parser])

    group = parser.add_argument_group('Warming level matching')
    group.add_argument("--running-mean-window", default=CONFIG["preprocessing.running_mean_window"], help="default: %(default)s years")
    group.add_argument("--warming-level-file", default=None)
    group.add_argument("--warming-levels", type=float, default=CONFIG["preprocessing.quantilemap_warming_levels"], nargs='+', help="default: %(default)s")
    group.add_argument("--quantile-bins", default=CONFIG["preprocessing.quantilemap_quantile_bins"], type=int)

    group = parser.add_argument_group('Indicator variable')
    group.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    group.add_argument("-i", "--indicator", nargs='+', default=[], choices=CONFIG["indicator"], help="includes additional, secondary indicator with specific monthly statistics")
    group.add_argument("--season", nargs="+", default=list(CONFIG["preprocessing.seasons"]), choices=list(CONFIG["preprocessing.seasons"]))
    group.add_argument("--simulation-round", nargs="+", default=CONFIG["isimip.simulation_round"], help="default: %(default)s")
    group.add_argument("--projection-baseline", default=CONFIG["preprocessing.projection_baseline"], type=int, nargs=2, help="default: %(default)s")

    parser.add_argument("-O", "--overwrite", action='store_true')

    # group = parser.add_argument_group('Result')
    # group.add_argument("--backend", nargs="+", default=CONFIG["preprocessing.isimip_binned_backend"], choices=["csv", "feather"])
    # group.add_argument("-O", "--overwrite", action='store_true')
    # group.add_argument("--cpus", type=int)

    o = parser.parse_args()

    CONFIG["isimip.simulation_round"] = o.simulation_round
    CONFIG["preprocessing.projection_baseline"] = o.projection_baseline

    if o.warming_level_file is None:
        o.warming_level_file = get_warming_level_file(**{**CONFIG, **vars(o)})

    warming_levels = pd.read_csv(o.warming_level_file)
    quantilemap_warming_levels = np.asarray(o.warming_levels)
    warming_levels = warming_levels[warming_levels["warming_level"].isin(quantilemap_warming_levels)]

    root_dir = Path(o.warming_level_file).parent


    for name in o.variable + o.indicator:
        indicator = Indicator.from_config(name)

        for season in o.season:
            if indicator.frequency == "annual" and season != "annual":
                continue
            filepath = get_filepath(indicator.name, season, root_dir=root_dir)
            if filepath.exists() and not o.overwrite:
                logger.info(f"{filepath} already exists. Use -O or --overwrite to reprocess.")
                continue
            filepath.parent.mkdir(parents=True, exist_ok=True)

            array = make_quantile_map_array(indicator,
                                            warming_levels,
                                            season=season,
                                            quantile_bins=o.quantile_bins,
                                            running_mean_window=o.running_mean_window,
                                            projection_baseline=o.projection_baseline,
                                            )

            logger.info(f"Write to {filepath}")
            encoding = {array.name: {'zlib': True}}
            array.to_netcdf(filepath, encoding=encoding)


if __name__ == "__main__":
    main()