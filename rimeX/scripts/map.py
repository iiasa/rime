"""Map emulator"""
from pathlib import Path
import argparse
import glob
import xarray as xa
import numpy as np
import pandas as pd

from rimeX.datasets import get_datapath

from rimeX.scripts.share import (
    _get_gmt_parser, 
    validate_iam_filter,
    # _get_gmt_dataframe, 
    _get_gmt_ensemble, 
    log_parser,
    config_parser,
    logger,
    get_datapath,
    setup_logger,
    CONFIG,
    )


def main():
    gmt_parser = _get_gmt_parser(ensemble=False)

    parser = argparse.ArgumentParser(epilog="""""", 
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        parents=[log_parser, config_parser, gmt_parser])
    
    group = parser.add_argument_group('Impact indicator')
    group.add_argument("-v", "--variable", required=True)
    group.add_argument("--gwl-dim", default='gwl')
    group.add_argument("--gwl-values", type=float, nargs='*')
    group.add_argument("-i", "--impact-file", nargs='+', required=True,
        help=f'NetCDF file with lon and lat dimensions.')
    group.add_argument("--impact-filter", nargs='+', metavar="KEY=VALUE", type=validate_iam_filter, default=[],
        help="other fields e.g. --impact-filter scenario='ssp2*'")
    group.add_argument("--bbox", nargs=4, type=float, help='bounding box left right bottom top')

    group = parser.add_argument_group('Result')
    group.add_argument("-O", "--overwrite", action='store_true', help='overwrite final results')
    group.add_argument("-o", "--output-file", required=True)
    parser.add_argument("--pyam", action="store_true", help='use pyam instead of own wrapper')

    o = parser.parse_args()
    setup_logger(o)

    if not o.overwrite and Path(o.output_file).exists():
        logger.info(f"{o.output_file} already exists")
        return 

    # if o.year is None or len(o.year) != 1:
    #     logger.error(f"Expected one year. Got: {o.year}. Use the --year parameter")

    gmt_ensemble = _get_gmt_ensemble(o, parser)

    if gmt_ensemble.shape[1] > 1:
        logger.warning(f"{gmt_ensemble.shape[1]} columns found for GMT ensemble. Use the median.")
        gmt_ensemble = gmt_ensemble.median(axis=1)
    gmt_ensemble = gmt_ensemble.iloc[:, 0]

    files = sorted(sum([sorted(glob.glob(f)) or sorted(glob.glob(str(get_datapath(f)))) for f in o.impact_file], []))
    if not files:
        logger.error(f"No file found: {o.impact_file}")
        parser.exit(1)

    ds = xa.open_mfdataset(files, combine="nested", concat_dim=o.gwl_dim)

    if o.gwl_values is not None:
        ds.coords[o.gwl_dim] = o.gwl_values

    logger.debug(f"Impact dataset: {ds}")

    # Note the default is [0, 1, ..., n] (integers) in case the value is not already present, so test for float.
    if isinstance(ds[o.gwl_dim][0].item(), np.floating):
        logger.error(f"Expected floating type for {o.gwl_dim}. Got: {repr(ds[o.gwl_dim][0].item())}: {type(ds[o.gwl_dim][0].item())}")
        parser.exit(1)

    data = ds[o.variable]

    if len(data[o.gwl_dim]) == 1:
        logger.error("Only one global-warming level was passed.")
        parser.exit(1)

    if o.bbox:
        l, r, b, t = o.bbox
        data = data.sel(lon=slice(l, r), lat=slice(b, t) if data.lat[1] > data.lat[0] else slice(t, b))
    # data = data.squeeze()

    logger.info(f"Interpolate the data to {len(gmt_ensemble)} values")
    time_dim = gmt_ensemble.index.name or "year"
    maps = data.interp({o.gwl_dim : gmt_ensemble.values})
    # maps = maps.assign_coords({time_dim: gmt_ensemble.index.values})
    maps = maps.to_dataset()
    maps[time_dim] = (o.gwl_dim, gmt_ensemble.index.values)
    maps = maps.set_coords(time_dim).swap_dims({o.gwl_dim: time_dim})

    ds.close()

    logger.info(f"Write to {o.output_file}")
    Path(o.output_file).parent.mkdir(parents=True, exist_ok=True)
    maps.to_netcdf(o.output_file)

if __name__ == "__main__":
    main()