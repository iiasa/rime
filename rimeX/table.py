"""A faster emulator, or interpolator, that does not attempt to compute uncertainties.
"""
import argparse
import os
import Path
from rimeX.emulator import (recombine_gmt_table, 
    _get_gmt_parser, _get_gmt_dataframe, setup_logger, _get_impact_parser, _get_impact_data, log_parser, config_parser)

def main():
    gmt_parser = _get_gmt_parser(parser)
    impact_parser = _get_impact_parser(parser)
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, gmt_parser, impact_parser])
    parser.add_argument("--method", choices=["nearest", "interp"], default="nearest")
    parser.add_argument("--backend", nargs='+', default=['csv'], choices=['csv', 'netcdf'])
    parser.add_argument("-o", "--output-file", required=True)
	o = parser.parse_args()
    setup_logger(o)

    gmt_table = _get_gmt_dataframe(o, parser)
    impact_table = _get_impact_dataframe(o, parser)

    data = recombine_gmt_table(impact_data, gmt_table, method=o.method, return_dataarray=True)

    Path(o.backend).parent.mkdir(exist_ok=True, parents=True)

    if "netcdf" in o.backend:
        file = o.output_file if len(o.backend) == 1 else os.path.splitext(o.output_file)[0] + ".csv"
        logger.info(f"Write to {file}")
        data.to_csv(file)

    if "csv" in o.backend:
        file = o.output_file if len(o.backend) == 1 else os.path.splitext(o.output_file)[0] + ".nc"
        logger.info(f"Write to {file}")
        data.to_netcdf(file)


if __name__ == "__main__":
    main()