"""
Write files like:

/mnt/PROVIDE/climate_impact_explorer/data2/isimip_regional_data/FRA/latWeight/canesm5_historical_pr_fra_latweight.csv

Performance test
----------------
Take about 6 min per (model, experiment, variable) or 3 min for historical, for 762 different region-weights masks
So about 25 min per (model, variable).
There are 10 models, so less than 5 h to run on a single core.
Memory usage is moderate (5G VIRT, 3G RES), mostly due to pre-loading all (partial) masks.
"""
from pathlib import Path
import argparse
import glob
import tqdm
from itertools import groupby, product
import warnings
import numpy as np
import pandas as pd
import xarray as xa

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser
from rimeX.datasets.download_isimip import get_models, get_experiments, get_variables, isimip_parser


def get_files(variable, model, experiment, realm="*", domain="global", frequency=None, member="*", obs="*", year_start="*", year_end="*", root=None, simulation_round=None):
    if root is None: root = CONFIG["isimip.download_folder"]
    if simulation_round is None: simulation_round = CONFIG["isimip.simulation_round"]
    frequency = frequency or CONFIG.get(f"indicator.{variable}.frequency", "monthly")
    model_lower = model.lower()
    model_upper = {m.lower():m for m in get_models()}[model_lower] if (model_lower and model_lower != "*") else "*"
    input_data = "*"
    if "ISIMIP2" in simulation_round:
        frequency2 = {"annual": "year", "monthly":"month", "daily": "day"}.get(frequency, frequency)
        member = "*"
        pattern = root + "/" + f"{simulation_round}/{input_data}/{realm}/biascorrected/{domain}/{experiment}/{model_upper}/{variable}_{frequency2}_{model_upper}_{experiment}_{member}_*_{year_start}0101-{year_end}1231.nc4"
    else:
        pattern = root + "/" + f"{simulation_round}/{input_data}/climate/{realm}/bias-adjusted/{domain}/{frequency}/{experiment}/{model_upper}/{model_lower}_{member}_{obs}_{experiment}_{variable}_{domain}_{frequency}_{year_start}_{year_end}.nc"
    return sorted(glob.glob(pattern))


def get_regional_averages_file(variable, model, experiment, region, weights, simulation_round=None, root=None):
    if simulation_round is None: simulation_round = CONFIG["isimip.simulation_round"]
    # if root is None: root = Path(CONFIG["isimip.climate_impact_explorer"]) / simulation_round
    if root is None: root = Path(CONFIG["isimip.climate_impact_explorer"]) / {"ISIMIP2b":"isimip2", "ISIMIP3b":"isimip3"}.get(simulation_round, simulation_round)
    return Path(root) / f"isimip_regional_data/{region}/{weights}/{model.lower()}_{experiment}_{variable}_{region.lower()}_{weights.lower()}.csv"


def get_coords(res=0.5):
    lon = np.arange(-180 + res/2, 180, res)
    lat = np.arange(90 - res/2, -90, -res)
    return lon, lat


def get_region_mask(region, weights, masks_folder=None):
    """return DataArray mask from a subregion"""
    if masks_folder is None: masks_folder = CONFIG["preprocessing.regional.masks_folder"]
    path = Path(masks_folder) / f"{region}/masks/{region}_360x720lat89p75to-89p75lon-179p75to179p75_{weights}.nc4"
    with xa.open_dataset(path) as ds:
        # return ds[region].load()
        return ds.load()
        # return m.reindex(lon=np.arange(), )

def _regional_average(v, mask):
    """Country averages

    v: numpy array with last dims (..., lat, lon)
    mask: numpy array defined on the same grid as v

    Returns a numpy array
    """
    # Determines the model `mask` (assuming it is constant throughout)
    # Equivalent to: ~np.isnan(v[0,...,0,:,:])
    valid_first_slice = np.isfinite(v[tuple(0 for _ in range(v.ndim - 2))])
    m = np.isfinite(mask) & (mask > 0) & valid_first_slice

    if not m.any():
        logger.debug(f"no valid data")

    weights = mask[m]
    return (v[..., m] * weights).sum(axis=-1) / weights.sum()


def preload_masks(regions, weights):

    masks = {}
    for weights in weights:
        for region in regions:
            try:
                masks[(region, weights)] = get_region_mask(region, weights)

            except FileNotFoundError as error:
                # logger.warning(str(error))
                logger.warning(f"No mask found for {region} {weights}")

    return masks


def main():

    ALL_MASKS = sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    parser.add_argument("-v", "--variable", nargs='+', choices=get_variables(), required=True)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    group.add_argument("--region", nargs='+', default=ALL_MASKS, choices=ALL_MASKS)
    group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])

    o = parser.parse_args()
    setup_logger(o)

    masks = preload_masks(o.region, o.weights)

    for variable in o.variable:
        for model in o.model:
            for experiment in o.experiment:

                todo = [(region, weights) for region in o.region for weights in o.weights if o.overwrite or not get_regional_averages_file(variable, model, experiment, region, weights).exists()]

                if not todo:
                    logger.info(f"{variable}, {model}, {experiment} region-mask averages already exist")
                    continue

                elif len(todo) < len(o.region)*len(o.weights):
                    logger.info(f"{variable}, {model}, {experiment}:: {len(todo)} / {len(o.region)*len(o.weights)} region-mask left to process")

                else:
                    logger.info(f"{variable}, {model}, {experiment}:: process {len(todo)} region-mask")

                results = {}

                for file in tqdm.tqdm(get_files(variable, model, experiment, frequency=o.frequency, simulation_round=o.simulation_round)):
                    with xa.open_dataset(file) as ds:

                        v = ds[variable].load()
                        assert v.dims == ("time", "lat", "lon") # no need to be more general here

                        # silence some warnings that may occur
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', r'invalid value encountered in divide')
                            warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')

                            for region, weights in todo:

                                # save intermediary file in the ISIMIP tree structure (useful if time-slices are added later, e.g. historical back in time)
                                tag = f"region-{region}-{weights}"
                                filetmp = Path(file.replace("global", tag))

                                if filetmp.exists():
                                # if filetmp.exists() and not o.overwrite: # uncomment if the region itself if redefined
                                    res = xa.open_dataset(filetmp)[variable].load()

                                else:
                                    if (region, weights) not in masks:
                                        continue

                                    ds_mask = masks[(region, weights)]
                                    v_extract = v.reindex(lon=ds_mask.lon.values, lat=ds_mask.lat.values)

                                    subregions = list(ds_mask)

                                    res = xa.DataArray(
                                        np.array([_regional_average(v_extract.values, ds_mask[k].values) for k in subregions]).T,
                                        name=variable, dims=("time", "region"), coords={"time": ds.time, "region": subregions})

                                    filetmp.parent.mkdir(exist_ok=True, parents=True)
                                    res.to_netcdf(filetmp, encoding={variable: {"zlib": True}})

                                results[(file, region, weights)] = res

                        # clean-up memory
                        del v

                # recombine by year and write to disk
                rw_key = lambda r: (r[0][1], r[0][2])
                for (region, weights), rw_group in groupby(sorted(results.items(), key=rw_key), key=rw_key):
                    result = xa.concat([data for key, data in rw_group], dim="time")
                    assert type(result) is xa.DataArray
                    ofile = get_regional_averages_file(variable, model, experiment, region, weights)

                    logger.info(f"Write region average output {ofile}")
                    ofile.parent.mkdir(exist_ok=True, parents=True)
                    # result.to_dataset(name=o.variable).to_netcdf(ofile)
                    result.to_pandas().to_csv(ofile)


if __name__ == "__main__":
    main()