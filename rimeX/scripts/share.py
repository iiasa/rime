"""
For a given scenario, return the mapped percentile for an indicator
"""
from pathlib import Path
import argparse
import glob
import numpy as np
import pandas as pd
import xarray as xa

import rimeX
from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser

from rimeX.compat import FastIamDataFrame, concat, read_table
from rimeX.compat import homogenize_table_names, load_files
from rimeX.datasets import get_datapath


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
    group.add_argument("--gsat-file", nargs="*", help='pyam-readable data')
    group.add_argument("--gsat-variable", nargs="+", default="*GSAT*", help="Filter iam variable")
    group.add_argument("--gsat-scenario", nargs="+", help="Filter iam scenario e.g. --gsat-scenario SSP1.26")
    group.add_argument("--gsat-model", nargs="+", help="Filter iam model")
    group.add_argument("--gsat-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --gsat model='IMAGE 3.0.1' scenario=SSP1.26", action='append')
    group.add_argument("--gsat-index", help="specify what defines a unique 'row'.")
    group.add_argument("--year", type=int, nargs="*", help="specify a set of years (e.g. for maps)")

    group.add_argument("--projection-baseline", type=int, nargs=2, default=CONFIG['emulator.projection_baseline'])
    group.add_argument("--projection-baseline-offset", type=float, default=CONFIG['emulator.projection_baseline_offset'])


    # group = parser.add_argument_group('Ensemble' + ' (invalid for this command)' if not ensemble else '')
    group = parser.add_argument_group('Ensemble' if ensemble else argparse.SUPPRESS)
    group.add_argument("--gsat-resample", action="store_true", help="Fit a distribution to GSAT from which to resample")
    group.add_argument("--gsat-dist", default="auto",
        choices=["auto", "norm", "lognorm"],
        # choices=["auto", "norm", "lognorm"],
        help="In auto mode, a normal or log-normal distribution will be fitted if percentiles are provided")
    group.add_argument("--gsat-samples", default=100, type=int, help="GSAT samples to draw if --gsat-fit is set")

    group.add_argument("--time-step", type=int, help="GSAT time step. By default whatever time-step is present in the input file.")
    group.add_argument("--save-gsat", help='filename to save the processed GSAT (e.g. for debugging)')

    group.add_argument("--magicc-files", nargs='+', help='if provided these files will be used instead if iam scenario')

    # group.add_argument("--no-check-single-index", action='store_false', dest='check_single_index', help=argparse.SUPPRESS)
    group.add_argument("--check-single-index", action='store_true', dest='check_single_index', help=argparse.SUPPRESS)

    if "pyam" not in [a.dest for a in parser._actions]:
        parser.add_argument("--pyam", action="store_true", help='use pyam instead of own wrapper')


    if not ensemble:
        for action in group._actions:
            action.help = argparse.SUPPRESS
        
    return parser


def _get_gmt_dataframe(o, parser):

    assert not o.magicc_files

    if not o.gsat_file:
        parser.error("Need to indicate MAGICC or IAM data file --gsat-file")
        parser.exit(1)
            
    iamdf_filtered = load_files(o.gsat_file, and_filters={
        "variable":o.gsat_variable, 
        "model": o.gsat_model, 
        "scenario": o.gsat_scenario
        }, or_filters=o.gsat_filter,
        index=o.gsat_index)

    if len(iamdf_filtered) == 0:
        logger.error(f"0-length dataframe")
        parser.exit(1)

    if o.check_single_index:
        if len(iamdf_filtered.index) > 1:
            logger.warning(f"More than one index")
            logger.warning(f"Remaining index: {str(iamdf_filtered.index)}")
            # parser.exit(1)

        if not o.gsat_resample and len(iamdf_filtered.variable) > 1:
            logger.warning(f"More than one variable")
            logger.warning(f"Remaining variable: {str(iamdf_filtered.variable)}")
            # parser.exit(1)

        if not o.gsat_resample and len(iamdf_filtered) != len(iamdf_filtered.year):
            logger.warning(f"More entries than years. Years: {len(iamdf_filtered.year)}. Entries: {len(iamdf_filtered)}")
            logger.warning(f"E.g. entries for first year:\n{str(iamdf_filtered.filter(year=iamdf_filtered.year[0]).as_pandas())}")
            # parser.exit(1)

    return iamdf_filtered.as_pandas()


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
            from rimeX.stats import fit_dist, repr_dist

            logger.info(f"Fit GSAT temperature distribution ({o.gsat_dist}) with {o.gsat_samples} samples.")

            if len(df.variable.unique()) != 3:
                logger.error(f"Expected three variables in GSAT fit mode. Found {len(df.variable.unique())}")
                logger.error(f"Remaining variable: {str(df.variable.unique())}")
                parser.exit(1)

            if len(df) != len(df.year.unique())*3:
                logger.error(f"Number of entries expected: 3 * years. Got {len(df)} entries and {len(df.year.unique())} years.")
                logger.error(f"E.g. entries for first year:\n{str(iamdf_filtered.filter(year=df.year[0]).as_pandas())}")
                parser.exit(1)                

            try:
                sat_quantiles = _sort_out_quantiles(df.variable.unique())
            except Exception as error:
                logger.error(f"Failed to extract quantiles.")
                logger.error(f"Expected variables contained the following strings: {dict(QUANTILES_MAP)}")
                logger.error(f"Remaining variables: {str(df.variable.unique())}")
                parser.exit(1)

            gmt_q = pd.DataFrame({q: df[df["variable"] == sat_quantiles[q]].set_index('year')['value'] for q in [50, 5, 95]})

            # Fit & resample
            nt = gmt_q.shape[0]
            ens = np.empty((nt, o.gsat_samples))
            for i in range(nt):
                # fit
                quants = [50, 5, 95]
                dist = fit_dist(gmt_q.iloc[i][quants], quants, o.gsat_dist)
                logger.debug(f"{i}: {repr_dist(dist)}")

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
            logger.info(f"Interpolate GSAT to {o.time_step}-year(s) time-step...")
            years = np.arange(gmt_ensemble.index[0], gmt_ensemble.index[-1]+o.time_step, o.time_step)
            gmt_ensemble = xa.DataArray(gmt_ensemble.values, coords={"year": gmt_ensemble.index}, dims=['year', 'sample']).interp(year=years).to_pandas()
            logger.info(f"Interpolate GSAT to {o.time_step}-year(s) time-step...done")


    if o.save_gsat:
        logger.info("Save GSAT...")
        gmt_ensemble.to_csv(o.save_gsat)
        logger.info("Save GSAT...done")

    return gmt_ensemble


def _get_impact_parser():

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Impact indicator')
    group.add_argument("-v", "--variable", nargs="*")
    group.add_argument("--region")
    # group.add_argument("--format", default="ixmp4", choices=["ixmp4", "cie"])
    group.add_argument("--impact-file", nargs='+', default=[], 
        help=f'Files such as produced by Werning et al 2014 (.csv with ixmp4 standard). Also accepted is a glob * pattern to match downloaded datasets (see also rime-download-ls).')
    group.add_argument("--impact-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --impact-filter scenario='ssp2*'", action="append")
    group.add_argument("--impact-index")
    group.add_argument("--model", nargs="+", help="if provided, only consider a set of specified model(s)")
    group.add_argument("--scenario", nargs="+", help="if provided, only consider a set of specified experiment(s)")
    # group.add_argument("--pyam", action="store_true", help='use pyam instead of own wrapper')
    # already added to GMT parser...

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

    impact_data_table = load_files(o.impact_file, and_filters={
        "variable":o.variable, 
        "model": o.model, 
        "scenario": o.scenario,
        "region": o.region,
        }, or_filters=o.impact_filter,
        index=o.impact_index)

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