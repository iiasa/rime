"""A faster emulator, or interpolator, that does not attempt to compute uncertainties.
"""
import argparse
import os
from pathlib import Path
import xarray as xa
from rimeX.emulator import recombine_gmt_table
from rimeX.compat import _get_ssp_mapping

from rimeX.scripts.share import (
    _get_gmt_parser, 
    _get_impact_parser, 
    _get_impact_data, 
    _get_gmt_dataframe, 
    log_parser,
    config_parser,
    logger,
    get_datapath,
    setup_logger,
    )

def main():
    gmt_parser = _get_gmt_parser()
    impact_parser = _get_impact_parser()
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, gmt_parser, impact_parser])
    parser.add_argument("--method", choices=["nearest", "linear"], default="linear")
    parser.add_argument("--bounds-check", action="store_true")
    parser.add_argument("--nc-impacts", nargs='+')
    parser.add_argument("-o", "--output-file", required=True)
    parser.add_argument("--ignore-year", action='store_true')
    parser.add_argument("--ignore-ssp", action='store_true')
    
    o = parser.parse_args()
    setup_logger(o)

    gmt_table = _get_gmt_dataframe(o, parser)

    if o.nc_impacts:
        assert not o.impact_file, 'cannot have both --nc-impact and --impact-file'
        impact_data = xa.open_mfdataset([get_datapath(f) for f in o.nc_impacts])

    else:        
        impact_data = _get_impact_data(o, parser)

    data = recombine_gmt_table(impact_data, gmt_table, method=o.method, bounds_error=o.bounds_check, 
        ignore_year=o.ignore_year, ignore_ssp=o.ignore_ssp)

    Path(o.output_file).parent.mkdir(exist_ok=True, parents=True)

    # if "csv" in o.backend:
    file = o.output_file
    logger.info(f"Write to {file}")
    data.to_csv(file, index=None)


if __name__ == "__main__":
    main()