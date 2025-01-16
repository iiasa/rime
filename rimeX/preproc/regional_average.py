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
import os
from pathlib import Path
import argparse
import concurrent.futures
import glob
import tqdm
from itertools import groupby, product
import warnings
import math
import numpy as np
import pandas as pd
import xarray as xa

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser
from rimeX.datasets.download_isimip import get_models, get_experiments, get_variables, isimip_parser
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.compat import open_dataset, open_mfdataset
from rimeX.tools import dir_is_empty


def get_files(variable, model, experiment, **kwargs):
    indicator = Indicator.from_config(variable)
    try:
        return [indicator.get_path(experiment, model, **kwargs)]
    except ValueError:
        return []

def get_regional_averages_file(variable, model, experiment, region, weights, impact_model=None, **kwargs):
    indicator = Indicator.from_config(variable)
    return indicator.get_path(experiment, model, region=region, regional_weight=weights, model=impact_model, **kwargs)
    # except ValueError:
    #     return []

def get_coords(res=0.5):
    lon = np.arange(-180 + res/2, 180, res)
    lat = np.arange(90 - res/2, -90, -res)
    return lon, lat


def get_mask_file(region, weights, masks_folder=None):
    if masks_folder is None: masks_folder = CONFIG["preprocessing.regional.masks_folder"]
    return Path(masks_folder) / f"{region}/masks/{region}_360x720lat89p75to-89p75lon-179p75to179p75_{weights}.nc4"

def open_region_mask(region, weights, masks_folder=None):
    """return DataArray mask from a subregion"""
    path = get_mask_file(region, weights, masks_folder)
    return xa.open_dataset(path)

_MASKS_CACHE = {}

def get_region_masks(region, weights, masks_folder=None):
    if (region, weights) not in _MASKS_CACHE:
        _MASKS_CACHE[(region, weights)] = open_region_mask(region, weights, masks_folder)
    return _MASKS_CACHE[(region, weights)]

def get_all_regions():
    return sorted(o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*") if not dir_is_empty(o) and not dir_is_empty(o / "masks"))

def _regional_average(v, mask):
    """Country averages

    v: numpy array with last dims (..., lat, lon)
    mask: numpy array defined on the same grid as v

    Returns a numpy array
    """
    # Determines the model `mask` (assuming it is constant throughout)
    # Equivalent to: ~np.isnan(v[0,...,0,:,:])
    first_slice = v[tuple(0 for _ in range(v.ndim - 2))]
    valid_first_slice = np.isfinite(first_slice) & (np.abs(first_slice) < 1e10)

    m = np.isfinite(mask) & (mask > 0) & valid_first_slice

    if not m.any():
        logger.debug(f"no valid data")

    weights = mask[m]
    return (v[..., m] * weights).sum(axis=-1) / weights.sum()


def _calc_regional_averages_unfiltered(v, ds_mask, name=None, reindex=True):
    """Transform a DataArray time x lat x lon into a time x region dataset

    v : xarray.DataArray (will be reindexed onto ds_mask)
    ds_mask : xarray.Dataset of binary masks (the regions)
    """
    assert v.dims == ("time", "lat", "lon") # no need to be more general here

    if reindex:
        # at the time of writing, the mask dataset is defined on a local grid
        # so we need to extract the larger dataset onto that local grid
        v = v.reindex(lon=ds_mask.lon.values, lat=ds_mask.lat.values)

    subregions = list(ds_mask)

    # first_slice = v[tuple(0 for _ in range(v.ndim - 2))]
    # valid_first_slice = np.isfinite(first_slice) & (np.abs(first_slice) < 1e10)
    valid = np.isfinite(v.values) & (np.abs(v.values) < 1e10)
    valid_any_slice = valid.any(axis=0)
    valid_all_slice = valid.all(axis=0)

    if np.any(valid_any_slice & ~valid_all_slice):
        logger.warning(f"{name or v.name} | {subregions[0]}... | Data available varies in time. Proceed to per-year regional average.")
        a = np.full((v.shape[0], len(subregions)), np.nan)
        for j, k in enumerate(subregions):
            mask = ds_mask[k].values
            for i in range(v.shape[0]):
                a[i:i+1, j] = _regional_average(v.values[i:i+1], mask)
        return xa.DataArray(a, name=name, dims=("time", "region"), coords={"time": v.time, "region": subregions})

    return xa.DataArray(
        np.array([_regional_average(v.values, ds_mask[k].values) for k in subregions]).T,
        name=name, dims=("time", "region"), coords={"time": v.time, "region": subregions})


def calc_regional_averages(v, ds_mask, name=None, **kwargs):
    """Transform a DataArray time x lat x lon into a time x region dataset

    v : xarray.DataArray (will be reindexed onto ds_mask)
    ds_mask : xarray.Dataset of binary masks (the regions)
    """
    # silence some warnings that may occur
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')

        return _calc_regional_averages_unfiltered(v, ds_mask, name=name, **kwargs)


def open_map_files(indicator, simus, **isel):
    files = [indicator.get_path(**simu) for simu in simus]
    ds = open_mfdataset(files, combine='nested', concat_dim="time", **isel)
    return ds[indicator.check_ncvar(ds)]

def get_all_subregion(region, weights="latWeight"):
    with open_region_mask(region, weights) as mask:
        return list(mask)

def _dataframe_to_dataarray(df, **kwargs):
    return xa.DataArray(df,
            # make sure we have dates as index (and not just years, cause the calling function needs dates)
            coords=[pd.to_datetime(df.index.astype(str)), df.columns],
            dims=["time", "region"], **kwargs,
            )

def _open_regional_data(indicator, simu, regions=None, weights="latWeight", admin=False, **kwargs):
    """This function loads data from the CSV files as computed in the preprocessing step
    """
    if regions is None:
        regions = get_all_regions()

    # if admin averages are not required, load the regional averages excluding the admin boundaries,
    # which are in a sperate file for convenience
    if not admin:
        file = indicator.get_path(**simu, regional=True, regional_weight=weights)
        df = pd.read_csv(file, index_col=0)
        return _dataframe_to_dataarray(df, name=indicator.ncvar)

    files = [indicator.get_path(**simu, region=region, regional_weight=weights, **kwargs)
                                    for region in regions]

    n0 = len(files)

    missing_regions = [region for (f, region) in zip(files, regions) if not f.exists()]
    missing_files = [f for f in files if not f.exists()]
    files = [f for f in files if f.exists()]
    if len(files) == 0:
        print("missing files", missing_files)
        raise FileNotFoundError(f"No regional files found for {indicator.name} {simu['climate_forcing']} {simu['climate_scenario']} {simu.get('model')}")

    if len(files) < n0:
        logger.debug(f"Missing regions {missing_regions}")
        logger.debug(f"Only {len(files)} out of {n0} files exist. Skip the missing ones.")

    if admin:
        dfs = pd.concat([pd.read_csv(file, index_col=0) for file in files], axis=1)  # concat region and their admin boundaries
        # remove duplicate columns
        dfs = dfs.loc[:, ~dfs.columns.duplicated()]
    else:
        dfs = pd.concat([pd.read_csv(file, index_col=0).iloc[:, :1] for file in files], axis=1)  # only use the first column (full region)

    return _dataframe_to_dataarray(dfs, name=indicator.ncvar)


def open_regional_files(indicator, simus, **kwargs):
    return xa.concat([_open_regional_data(indicator, simu, **kwargs)
                      for simu in simus], dim="time") # historical and future


def open_files(indicator, simus, regional=False, isel={}, **kwargs):
    if regional:
        data = open_regional_files(indicator, simus, **kwargs)
    else:
        data = open_map_files(indicator, simus)
    if isel:
        data = data.isel(isel)
    return data

def _check_file(file):
    if not file.exists():
        return False
    if file.stat().st_size == 0:
        logger.warning(f"Empty file: {file} -> remove")
        file.unlink()
        return False
    return True

def _crunch_regional_averages(indicator, simu, o, write_merged_regional_averages=True):

    todo = [(region, weights) for region in o.region
            for weights in o.weights
                if get_mask_file(region, weights).exists()
                    and (o.overwrite or not _check_file(indicator.get_path(**simu, region=region, regional_weight=weights)))]
                            # and (o.overwrite or not get_regional_averages_file(variable, model, experiment, region, weights, impact_model=impact_model).exists())]

    # also consider merged files
    merged_files = [indicator.get_path(**simu, regional=True, regional_weight=weights) for weights in o.weights]
    missing_files = [f for f in merged_files if o.overwrite or not f.exists()]
    write_merged_regional_averages_ = write_merged_regional_averages and len(missing_files) > 0
    nothing_todo = not todo and not write_merged_regional_averages_

    if nothing_todo:
        logger.info(f"{indicator.name}, {simu}: all region-mask averages already exist")
        return

    elif not todo:
        logger.info(f"{indicator.name}, {simu}: compute merged regional averages: {missing_files}")

    elif 0 < len(todo) < len(o.region)*len(o.weights):
        logger.info(f"{indicator.name}, {simu}:: {len(todo)} / {len(o.region)*len(o.weights)} region-mask averages to process")

    else:
        logger.info(f"{indicator.name}, {simu}:: process {len(todo)} region-mask")


    # calculate regional averages including admin boundaries
    if todo:
        file = indicator.get_path(**simu)
        with open_dataset(file) as ds:

            v = ds[indicator.check_ncvar(ds)].load()

            if v.dtype.name.startswith("timedelta"):
                v.values = v.values.astype("timedelta64[D]").astype(float)

            for weight in o.weights:
                for region in o.region:

                    try:
                        mask = get_region_masks(region, weight)
                    except FileNotFoundError:
                        continue

                    ofile_csv = indicator.get_path(**simu, region=region, regional_weight=weight)
                    if not o.overwrite and ofile_csv.exists():
                        continue

                    # keep the same netCDF name as the original file, for consistency with lat/lon file
                    res = calc_regional_averages(v, mask, name=indicator.ncvar)

                    # write to CSV
                    ofile_csv.parent.mkdir(exist_ok=True, parents=True)
                    res.to_pandas().to_csv(ofile_csv)

            # clean-up memory
            del v

    # make a combined file with all regions but without admin boundaries
    if write_merged_regional_averages:
        for weight in o.weights:
            ofile = indicator.get_path(**simu, regional=True, regional_weight=weight)
            if not o.overwrite and ofile.exists():
                continue
            rfiles = [(indicator.get_path(**simu, region=region, regional_weight=weight), region)
                                for region in o.region]
            # some masks do not exist, so we need to filter unexistent files
            rfiles = [(f, region) for f, region in rfiles if f.exists()]

            # load all regional averages and keep only the first column (full region, no admin)
            # data = pd.concat([pd.read_csv(f, index_col=0)[region] for f, region in rfiles], axis=1)
            rdata = []
            for f, region in rfiles:
                try:
                    data = pd.read_csv(f, index_col=0)[region]
                except:
                    raise
                rdata.append(data)

            data = pd.concat(rdata, axis=1)
            data.name = indicator.ncvar
            data.index = pd.to_datetime(data.index)
            data.index.name = "time"

            logger.info(f"{indicator.name}|{simu}|{weight} : write merged regional averages to {ofile}")
            ofile.parent.mkdir(exist_ok=True, parents=True)
            data.to_csv(ofile)
            # data.to_netcdf(ofile, encoding={variable: {'zlib': True}})

            # clean-up memory
            del data


def main():

    ALL_REGIONS = get_all_regions()
    all_variables = list(CONFIG["isimip.variables"]) + sorted(set(v.split(".")[0] for v in CONFIG["indicator"]))
    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    # parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=all_variables, help="includes additional, secondary indicator with specific monthly statistics")
    parser.add_argument("--overwrite", action='store_true')
    # parser.add_argument("--backends", nargs="+", choices=["csv", "netcdf"], default=["netcdf", "csv"])
    parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    group.add_argument("--region", nargs='+', default=ALL_REGIONS, choices=ALL_REGIONS)
    group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--cpus", type=int)

    o = parser.parse_args()
    setup_logger(o)

    write_merged_regional_averages = True
    if list(o.region) != list(get_all_regions()):
        logger.warning("Skip writing the summary netCDF file with all country averages because the required regions are different from the default")
        write_merged_regional_averages = False

    def iterator():

        for variable in o.indicator:

            indicator = Indicator.from_config(variable)

            for simu in indicator.simulations:
                if not _matches(simu["climate_forcing"], o.model):
                    continue
                if not _matches(simu["climate_scenario"], o.experiment):
                    continue
                if not _matches(simu.get("model"), o.impact_model):
                    continue

                yield indicator, simu


    all_items = [(indicator, simu) for indicator, simu in iterator()]

    if o.cpus is not None:
        o.cpus = min(o.cpus, len(all_items))

    if o.cpus and o.cpus > 1:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=o.cpus)
    else:
        # dummy Executor pool that calls the function directly and returns None
        pool = argparse.Namespace(submit=lambda f, *args, **kwargs: f(*args, **kwargs))
    jobs = []

    for indicator, simu in all_items:
        # _crunch_regional_averages(indicator, simu, masks, o, write_merged_regional_averages=write_merged_regional_averages)
        jobs.append( pool.submit(_crunch_regional_averages, indicator, simu, o, write_merged_regional_averages=write_merged_regional_averages) )

    for job in tqdm.tqdm(jobs):
        if job is not None:
            job.result()

if __name__ == "__main__":
    main()