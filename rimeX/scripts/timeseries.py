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

from rimeX.records import (
    make_equiprobable_groups, interpolate_years, interpolate_warming_levels, 
    fit_records, average_per_group, 
    pool_dims, drop_dims,
    )

from rimeX.emulator import recombine_gmt_ensemble
from rimeX.compat import _simplify, FastIamDataFrame

from rimeX.scripts.share import (
    _get_gmt_parser, 
    _get_impact_parser, 
    _get_impact_data, 
    _get_gmt_ensemble, 
    )


def main(cmd=None):

    gmt_parser = _get_gmt_parser(ensemble=True)
    impact_parser = _get_impact_parser()

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, 
        parents=[log_parser, config_parser, gmt_parser, impact_parser])
    
    group = parser.add_argument_group('Re-indexing, Interpolation & Resampling')
    # group.add_argument("--pool", type=_simplify, nargs='+', action="append", 
    group.add_argument("--pool", type=_simplify, nargs='+', default=[],
        help=f"Pool dimensions (ignore them in all grouping operations, just like metadata). The following will always be included: {CONFIG['index.ignore']}")
    group.add_argument("--average", nargs='+', action="append",
        help="""Average dimension(s). By default the dimension(s) are pooled before averaging.
If this behavior is not desired, input --average several times""")
    group.add_argument("--impact-resample", action="store_true", 
        help="""Fit a distribution to the impact data from which to resample. 
        Assumes the quantile variables are named "{NAME}|5th percentile" and "{NAME}|95th percentile".""")
    group.add_argument("--impact-dist", default="auto", 
        choices=["auto", "norm", "lognorm"], 
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--impact-samples", default=100, type=int, help="Number of samples to draw if --impact-fit is set")
    group.add_argument("--interp-warming-levels", action="store_true", 
        help=f"interpolate warming levels (set default step via --warming-level-step)")
    group.add_argument("--warming-level-step", type=float,
        default=CONFIG.get("emulator.warming_level_step"),
        help="Default step for --interp-warming-level (default: %(default)s)")
    group.add_argument("--interp-years", help=f"interpolate years (match input GSAT time-series)", action='store_true')

    group = parser.add_argument_group('Aggregation')
    # group.add_argument("--equiprobable-models", action="store_true", help="if True, each model will have the same probability")
    group.add_argument("--equiprobable-dims", nargs='+', type=_simplify,
        help="Make sure weights within a group are equal, e.g --equiprobable model. By default special dimensions like `variable`, `region` and `warming_level` are added to the group specifiers.")
    group.add_argument("--match-year-population", action="store_true")
    # group.add_argument("--method", default="ensemble", choices=["ensemble", "table"])

    group = parser.add_argument_group('Result')
    group.add_argument("--quantiles", nargs='+', default=CONFIG["emulator.quantiles"], help="(default: %(default)s)")
    group.add_argument("--no-overwrite", action='store_false', dest='overwrite', help=argparse.SUPPRESS)
    group.add_argument("-O", "--overwrite", action='store_true', help=argparse.SUPPRESS)
    group.add_argument("-o", "--output-file", required=True)
    group.add_argument("--save-impact-table", help='file name to save the processed impacts table (e.g. for debugging)')


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


    # All dimensions present in the dataset
    if o.index is None:
        all_impact_dims = [c for c in impact_data_frame if c not in CONFIG["index.ignore"] and c not in o.pool]

    else:
        all_impact_dims = o.index

        if "value" in all_impact_dims:
            raise ValueError(f"value cannot be in the index")
        if "warming_level" not in all_impact_dims:
            logger.info("add warming_level to index")
            all_impact_dims.append("warming_level")
        if o.match_year_population and "year" not in all_impact_dims:
            logger.info("add year to index")
            all_impact_dims.append("year")    

    if "warming_level" not in all_impact_dims:
        raise ValueError("warming_level not found")

    if o.match_year_population and "year" not in all_impact_dims:
        raise ValueError("year not found")

    if o.pool:
        if "value" in o.pool:
            raise ValueError("`value` cannot be pooled")
        drop_dims(impact_data_records, o.pool)
        for dim in o.pool:
            if dim in all_impact_dims:
                all_impact_dims.remove(dim)

    if o.average:
        for dims in o.average:
            logger.info(f"average across {dims}...")
            impact_data_records = average_per_group(impact_data_records, by=[c for c in all_impact_dims if c not in dims])
            logger.info(f"average across {dims}...done")
            # refresh dims list
            for dim in dims:
                if dim in all_impact_dims:
                    all_impact_dims.remove(dim)

    # Fit and resample impact data if required
    if o.impact_resample:
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...")
        try:
            impact_data_records = fit_records(impact_data_records, o.impact_samples, dist_name=o.impact_dist,
                by=[c for c in all_impact_dims if c not in ['variable', 'quantile', 'percentile']])
        except Exception as error:
            raise
            logger.error(str(error))
            parser.exit(1)
        logger.info(f"Fit Impact Percentiles ({o.impact_dist}) with {o.impact_samples} samples...done")
        # all_impact_dims.remove("quantile")
        all_impact_dims += ["sample"]  # append with new dimension name


    # Interpolate records
    if o.interp_warming_levels:
        logger.info("Impact data: interpolate warming levels...")
        impact_data_records = interpolate_warming_levels(impact_data_records, o.warming_level_step,
            by=[c for c in all_impact_dims if c not in "warming_level"])
            # by=["variable", "region", "model", "scenario", "year", "sample"])
        logger.info("Impact data: interpolate warming levels...done")


    # For population dataset the year can be matched to temperatrure time-series. It must be interpolated to yearly values first.
    if o.interp_years:
        logger.info("Impact data: interpolate years...")
        impact_data_records = interpolate_years(impact_data_records, gmt_ensemble.index, 
            by=[c for c in all_impact_dims if c != "year"])
            # by=["variable", "region", "model", "scenario", "warming_level", "sample"])
        logger.info("Impact data: interpolate years...done")

    if o.equiprobable_dims:
        if "variable" not in o.equiprobable_dims: o.equiprobable_dims.append("variable")
        if "region" not in o.equiprobable_dims: o.equiprobable_dims.append("region")
        if "warming_level" not in o.equiprobable_dims: o.equiprobable_dims.append("warming_level")
        logger.info(f"Normalization to give equal weight to each group by dimensions {o.equiprobable_dims}...")
        make_equiprobable_groups(impact_data_records, by=o.equiprobable_dims)
        logger.info(f"Normalization to give equal weight to each group by dimensions {o.equiprobable_dims}...done")

    if o.save_impact_table:
        logger.info("Save impact table...")
        pd.DataFrame(impact_data_records).to_csv(o.save_impact_table, index=None)
        logger.info("Save impact table...done")


    # Only use future values to avoid getting in trouble with the warming levels.
    gmt_ensemble = gmt_ensemble.loc[2015:]  

    assert np.isfinite(gmt_ensemble.values).all(), 'some NaN in MAGICC run'

    # Check Impact Table
    check_index = ["warming_level"]
    if o.match_year_population: check_index.append("year")
    logger.info(f"Impact data prior binning via warming_level / year:\n{FastIamDataFrame(pd.DataFrame(impact_data_records), index=check_index)}")

    # Recombine GMT ensemble with binned ISIMIP data
    quantiles = recombine_gmt_ensemble(impact_data_records, gmt_ensemble, o.quantiles, match_year=o.match_year_population)

    # GMT result to disk
    logger.info(f"Write output to {o.output_file}")
    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)
    quantiles.to_csv(o.output_file)


if __name__ == "__main__":
    main()