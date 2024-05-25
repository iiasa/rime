"""A faster emulator, or interpolator, that does not attempt to compute uncertainties.
"""
import argparse
import os
from pathlib import Path
import xarray as xa
from rimeX.emulator import (recombine_gmt_table, get_datapath, logger,
    _get_gmt_parser, _get_gmt_dataframe, setup_logger, _get_impact_parser, _get_impact_data, log_parser, config_parser)

def main():
    gmt_parser = _get_gmt_parser()
    impact_parser = _get_impact_parser()
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, gmt_parser, impact_parser])
    parser.add_argument("--method", choices=["nearest", "linear"], default="linear")
    parser.add_argument("--bounds-check", action="store_true")
    parser.add_argument("--backend", nargs='+', default=['csv'], choices=['csv', 'netcdf'])
    parser.add_argument("--nc-impacts", nargs='+')
    parser.add_argument("--pyam", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-o", "--output-file", required=True)
    
    o = parser.parse_args()
    setup_logger(o)

    gmt_table = _get_gmt_dataframe(o, parser)

    if o.nc_impacts:
        assert not o.impact_file, 'cannot have both --nc-impact and --impact-file'
        impact_data = xa.open_mfdataset([get_datapath(f) for f in o.nc_impacts])

    else:        
        impact_data = _get_impact_data(o, parser)

    data = recombine_gmt_table(impact_data, gmt_table, method=o.method, return_dataarray=True, bounds_error=o.bounds_check)

    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)

    if "csv" in o.backend:
        file = o.output_file if len(o.backend) == 1 else os.path.splitext(o.output_file)[0] + ".csv"
        logger.info(f"Write to {file}")
        data.to_series().reset_index(name='value').to_csv(file)

    if "netcdf" in o.backend:
        file = o.output_file if len(o.backend) == 1 else os.path.splitext(o.output_file)[0] + ".nc"
        logger.info(f"Write to {file}")
        data.reset_index('index').to_netcdf(file)


if __name__ == "__main__":
    main()