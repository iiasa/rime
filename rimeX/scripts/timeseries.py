"""
For a given scenario, return the mapped percentile for an indicator
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xa

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser

from rimeX.preproc.digitize import (
    make_equiprobable_groups, interpolate_years, interpolate_warming_levels, 
    fit_records, average_per_group)

from rimeX.emulator import recombine_gmt_ensemble

from rimeX.scripts.share import (
    _get_gmt_parser, 
    _get_impact_parser, 
    _get_impact_data, 
    _get_gmt_ensemble, 
    )

def main(cmd=None):

    gmt_parser = _get_gmt_parser(ensemble=True)
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
    group.add_argument("--warming-level-step", type=float,
        # default=CONFIG.get("preprocessing.warming_level_step"),
        help="Impact indicators will be interpolated to match this warming level (default: %(default)s)")
    group.add_argument("--impact-resample", action="store_true", 
        help="""Fit a distribution to the impact data from which to resample. 
        Assumes the quantile variables are named "{NAME}|5th percentile" and "{NAME}|95th percentile".""")
    group.add_argument("--impact-dist", default="auto", 
        choices=["auto", "norm", "lognorm"], 
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--impact-samples", default=100, type=int, help="Number of samples to draw if --impact-fit is set")

    group = parser.add_argument_group('Result')
    group.add_argument("--no-overwrite", action='store_false', dest='overwrite', help=argparse.SUPPRESS)
    group.add_argument("-O", "--overwrite", action='store_true', help=argparse.SUPPRESS)
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
        logger.info("average across scenarios...")
        impact_data_records = average_per_group(impact_data_records, by=("variable", "region", 'model', 'warming_level', 'year'))
        logger.info("average across scenarios...done")

    # Harmonize weights
    if o.equiprobable_models:
        logger.info("Normalization to give equal weight for each model per temperature bin...")        
        make_equiprobable_groups(impact_data_records, by=["variable", "region", "model", "warming_level"])
        logger.info("Normalization to give equal weight for each model per temperature bin...done")        

    # Fit and resample impact data if required
    if o.impact_resample:
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...")
        try:
            impact_data_records = fit_records(impact_data_records, o.impact_samples, dist_name=o.impact_dist,
                by=["region", "model", "scenario", "warming_level", "year"])
        except Exception as error:
            raise
            logger.error(str(error))
            parser.exit(1)
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...done")


    # Interpolate records
    if o.warming_level_step:
        logger.info("Impact data: interpolate warming levels...")
        impact_data_records = interpolate_warming_levels(impact_data_records, o.warming_level_step,
            by=["variable", "region", "model", "scenario", "year", "sample"])
        logger.info("Impact data: interpolate warming levels...done")


    # For population dataset the year can be matched to temperatrure time-series. It must be interpolated to yearly values first.
    if o.match_year_population:
        logger.info("Impact data: interpolate years...")
        impact_data_records = interpolate_years(impact_data_records, gmt_ensemble.index, 
            # by=['variable', "region", 'warming_level', "model", "scenario"])
            by=["variable", "region", "model", "scenario", "warming_level", "sample"])
        logger.info("Impact data: interpolate years...done")

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