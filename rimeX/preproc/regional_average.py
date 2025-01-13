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
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.compat import open_dataset, open_mfdataset

def get_files(variable, model, experiment, **kwargs):
    indicator = Indicator.from_config(variable)
    try:
        return [indicator.get_path(experiment, model, **kwargs)]
    except ValueError:
        return []

def get_regional_averages_file(variable, model, experiment, region, weights, **kwargs):
    indicator = Indicator.from_config(variable)
    try:
        return [indicator.get_path(experiment, model, region=region, regional_weight=weights, **kwargs)]
    except ValueError:
        return []

def get_coords(res=0.5):
    lon = np.arange(-180 + res/2, 180, res)
    lat = np.arange(90 - res/2, -90, -res)
    return lon, lat


def open_region_mask(region, weights, masks_folder=None):
    """return DataArray mask from a subregion"""
    if masks_folder is None: masks_folder = CONFIG["preprocessing.regional.masks_folder"]
    path = Path(masks_folder) / f"{region}/masks/{region}_360x720lat89p75to-89p75lon-179p75to179p75_{weights}.nc4"
    return xa.open_dataset(path)

def _regional_average(v, mask, _no_recursion=False, context=""):
    """Country averages

    v: numpy array with last dims (..., lat, lon)
    mask: numpy array defined on the same grid as v

    Returns a numpy array
    """
    # Determines the model `mask` (assuming it is constant throughout)
    # Equivalent to: ~np.isnan(v[0,...,0,:,:])
    first_slice = v[tuple(0 for _ in range(v.ndim - 2))]
    valid_first_slice = np.isfinite(first_slice) & (np.abs(first_slice) < 1e10)
    valid = np.isfinite(v)
    valid_any_slice = valid.any(axis=tuple(i for i in range(v.ndim - 2)))
    valid_all_slice = valid.all(axis=tuple(i for i in range(v.ndim - 2)))
    if np.any(valid_any_slice & ~valid_all_slice):
        logger.warning(f"{context} The nan mask may vary based on the time slice -> proceed to per-year regional average")
        assert _no_recursion is False, "_no_recursion should be False"
        a = np.full(v.shape[:-2], np.nan)
        # make things easier by reshaping the data
        import math
        a_flat = a.reshape(math.prod(a.shape[:-2]), *a.shape[-2:])
        v_flat = v.reshape(math.prod(v.shape[:-2]), *v.shape[-2:])
        for i in range(v_flat.shape[0]):
            a_flat[i:i+1] = _regional_average(v_flat[i:i+1], mask, _no_recursion=True)
        return a

    m = np.isfinite(mask) & (mask > 0) & valid_first_slice

    if not m.any():
        logger.debug(f"no valid data")

    weights = mask[m]
    return (v[..., m] * weights).sum(axis=-1) / weights.sum()


def get_all_regions():
    return sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])


def preload_masks(regions=None, weights=["latWeight"], admin=True):
    """Return a xarray.Dataset with all masks for the given regions and weights
    """
    if regions is None:
        regions = get_all_regions()
    masks = {}
    for weight in weights:
        for region in regions:
            try:
                with open_region_mask(region, weight) as mask:
                    mask_loaded = mask.load()
                    if not admin:
                        mask_loaded = mask_loaded[list(mask_loaded)[:1]]
                    masks[(region, weight)] = mask_loaded

            except FileNotFoundError as error:
                # logger.warning(str(error))
                logger.warning(f"No mask found for {region} {weight}")

    return masks


def preload_masks_merged(regions=None, weight="latWeight", admin=True):
    """Return a xarray.Dataset with all masks for the given regions and weights
    """
    if regions is None:
        regions = get_all_regions()
    masks = preload_masks(regions, [weight], admin)
    merged_masks = xa.merge(list(masks.values()), compat="override", combine_attrs="drop_conflicts", fill_value=np.nan)
    del merged_masks.coords["region"]
    return merged_masks


def get_merged_masks(regions, weights="latWeight", admin=True):
    key = (weights, len(regions) if len(regions) > 1 else regions[0], "admin" if admin else "noadmin")
    filepath = Path(CONFIG["indicators.folder"], "masks", "merged_"+"_".join(map(str, key))+".nc")
    if filepath.exists():
        logger.info(f"Load merged masks from {filepath}")
        return open_dataset(filepath)
    merged_masks = preload_masks_merged(regions, weights, admin)
    logger.info(f"Write merged masks to {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    merged_masks.to_netcdf(filepath, encoding={v: {'zlib': True} for v in merged_masks.data_vars})
    return merged_masks


def _calc_regional_averages_unfiltered(v, ds_mask, name=None, reindex=True):
    """Transform a DataArray time x lat x lon into a time x region dataset

    v : xarray.DataArray (will be reindexed onto ds_mask)
    ds_mask : xarray.Dataset of binary masks (the regions)
    """
    assert v.dims == ("time", "lat", "lon") # no need to be more general here

    if reindex:
        # at the time of writing, the mask dataset is defined on a local grid
        # so we need to extract the larger dataset onto that local grid
        v_extract = v.reindex(lon=ds_mask.lon.values, lat=ds_mask.lat.values)
    else:
        # for applications where the dataset and the mask are already on the same grid
        v_extract = v

    subregions = list(ds_mask)

    return xa.DataArray(
        np.array([_regional_average(v_extract.values, ds_mask[k].values, context=k) for k in subregions]).T,
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
    return open_mfdataset(files, combine='nested', concat_dim="time", **isel)[indicator.ncvar]

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




def main():

    ALL_MASKS = sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=CONFIG["indicator"], help="includes additional, secondary indicator with specific monthly statistics")
    parser.add_argument("--overwrite", action='store_true')
    # parser.add_argument("--backends", nargs="+", choices=["csv", "netcdf"], default=["netcdf", "csv"])
    parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    group.add_argument("--region", nargs='+', default=ALL_MASKS, choices=ALL_MASKS)
    group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])

    o = parser.parse_args()
    setup_logger(o)

    masks = preload_masks(o.region, o.weights)

    write_merged_regional_averages = True
    if list(o.region) != list(get_all_regions()):
        logger.warning("Skip writing the summary netCDF file with all country averages because the required regions are different from the default")
        write_merged_regional_averages = False


    for variable in o.variable+o.indicator:
        indicator = Indicator.from_config(variable)
        for simu in indicator.simulations:
            if not _matches(simu["climate_forcing"], o.model):
                continue
            if not _matches(simu["climate_scenario"], o.experiment):
                continue
            if not _matches(simu.get("model"), o.impact_model):
                continue

            todo = [(region, weights) for region in o.region
                    for weights in o.weights
                        if (region, weights) in masks
                            and (o.overwrite or not indicator.get_path(**simu, region=region, regional_weight=weights).exists())]
                                    # and (o.overwrite or not get_regional_averages_file(variable, model, experiment, region, weights, impact_model=impact_model).exists())]

            # also consider merged files
            merged_files = [indicator.get_path(**simu, regional=True, regional_weight=weights) for weights in o.weights]
            write_merged_regional_averages_ = write_merged_regional_averages and any(not f.exists() for f in merged_files)
            nothing_todo = not todo and write_merged_regional_averages_

            if nothing_todo:
                logger.info(f"{variable}, {simu}: all region-mask averages already exist")
                continue

            elif not todo:
                logger.info(f"{variable}, {simu}: compute merged regional averages")

            elif len(todo) < len(o.region)*len(o.weights):
                logger.info(f"{variable}, {simu}:: {len(todo)} / {len(o.region)*len(o.weights)} region-mask averages to process")

            else:
                logger.info(f"{variable}, {simu}:: process {len(todo)} region-mask")

            file = indicator.get_path(**simu)

            # calculate regional averages including admin boundaries
            with open_dataset(file) as ds:

                v = ds[indicator.ncvar].load()

                for weight in o.weights:
                    for region in o.region:

                        if not (region, weight) in masks:
                            continue

                        ofile_csv = indicator.get_path(**simu, region=region, regional_weight=weight)
                        if not o.overwrite and ofile_csv.exists():
                            continue

                        # keep the same netCDF name as the original file, for consistency with lat/lon file
                        res = calc_regional_averages(v, masks[(region, weight)], name=indicator.ncvar)

                        # write to CSV
                        ofile_csv.parent.mkdir(exist_ok=True, parents=True)
                        res.to_pandas().to_csv(ofile_csv)

                # clean-up memory
                del v

            # make a combined file with all regions but without admin boundaries
            if write_merged_regional_averages:
                for weight in o.weights:
                    rfiles = [(indicator.get_path(**simu, region=region, regional_weight=weight), region)
                                        for region in o.region]
                    # some masks do not exist, so we need to filter unexistent files
                    rfiles = [(f, region) for f, region in rfiles if f.exists()]

                    # load all regional averages and keep only the first column (full region, no admin)
                    data = pd.concat([pd.read_csv(f, index_col=0)[region] for f, region in rfiles], axis=1)
                    data.name = indicator.ncvar
                    data.index = pd.to_datetime(data.index)
                    data.index.name = "time"

                    ofile = indicator.get_path(**simu, regional=True, regional_weight=weight)
                    logger.info(f"{indicator.name}|{simu}|{weight} : write merged regional averages to {ofile}")
                    ofile.parent.mkdir(exist_ok=True, parents=True)
                    data.to_csv(ofile)
                    # data.to_netcdf(ofile, encoding={variable: {'zlib': True}})

                # clean-up memory
                del data

if __name__ == "__main__":
    main()