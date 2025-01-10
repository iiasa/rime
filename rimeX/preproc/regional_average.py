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


# def get_files(variable, model, experiment, realm="*", domain="global", frequency=None, member="*", obs="*", year_start="*", year_end="*", root=None, simulation_round=None):
#     if root is None: root = CONFIG["isimip.download_folder"]
#     if simulation_round is None: simulation_round = CONFIG["isimip.simulation_round"]
#     frequency = frequency or CONFIG.get(f"indicator.{variable}.frequency", "monthly")
#     model_lower = model.lower()
#     model_upper = {m.lower():m for m in get_models()}[model_lower] if (model_lower and model_lower != "*") else "*"
#     input_data = "*"
#     if "ISIMIP2" in simulation_round:
#         frequency2 = {"annual": "year", "monthly":"month", "daily": "day"}.get(frequency, frequency)
#         member = "*"
#         pattern = root + "/" + f"{simulation_round}/{input_data}/{realm}/biascorrected/{domain}/{experiment}/{model_upper}/{variable}_{frequency2}_{model_upper}_{experiment}_{member}_*_{year_start}0101-{year_end}1231.nc4"
#     else:
#         pattern = root + "/" + f"{simulation_round}/{input_data}/climate/{realm}/bias-adjusted/{domain}/{frequency}/{experiment}/{model_upper}/{model_lower}_{member}_{obs}_{experiment}_{variable}_{domain}_{frequency}_{year_start}_{year_end}.nc"
#     return sorted(glob.glob(pattern))
def get_files(variable, model, experiment, **kwargs):
    indicator = Indicator.from_config(variable)
    try:
        return [indicator.get_path(experiment, model, **kwargs)]
    except ValueError:
        return []

def get_regional_averages_file(variable, model, experiment, region, weights, simulation_round=None, root=None, impact_model=None):
    if simulation_round is None: simulation_round = CONFIG["isimip.simulation_round"]
    simulation_round = "-".join([{"isimip2b": "isimip2", "isimip3b": "isimip3"}.get(s.lower(), s.lower()) for s in simulation_round])
    if root is None: root = Path(CONFIG["isimip.climate_impact_explorer"]) / simulation_round
    if impact_model is not None:
        model = f"{model}_{impact_model}"
    return Path(root) / f"isimip_regional_data/{region}/{weights}/{model.lower()}_{experiment}_{variable}_{region.lower()}_{weights.lower()}.csv"

def get_coords(res=0.5):
    lon = np.arange(-180 + res/2, 180, res)
    lat = np.arange(90 - res/2, -90, -res)
    return lon, lat


def open_region_mask(region, weights, masks_folder=None):
    """return DataArray mask from a subregion"""
    if masks_folder is None: masks_folder = CONFIG["preprocessing.regional.masks_folder"]
    path = Path(masks_folder) / f"{region}/masks/{region}_360x720lat89p75to-89p75lon-179p75to179p75_{weights}.nc4"
    return xa.open_dataset(path)

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


def get_all_regions():
    return sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])

def preload_masks_merged(regions=None, weight="latWeight", admin=True):
    """Return a xarray.Dataset with all masks for the given regions and weights
    """
    if regions is None:
        regions = get_all_regions()
    merged = None
    for region in regions:
        try:
            with open_region_mask(region, weight) as mask:
                mask = mask.load()
        except FileNotFoundError as error:
            logger.warning(str(error))
            logger.warning(f"No mask found for {region} {weight}")
            continue
        if not admin:
            mask = mask[list(mask)[:1]] # only use the first column (full region)
        if merged is None:
            merged = mask
        else:
            for k in mask:
                # assert k not in merged, (k, list(merged))
                if k not in merged:
                    merged[k] = mask[k]
    return merged


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
                    masks[(region, weights)] = mask_loaded

            except FileNotFoundError as error:
                # logger.warning(str(error))
                logger.warning(f"No mask found for {region} {weight}")

    return masks


def _calc_regional_averages_unfiltered(v, ds_mask, name=None):
    """Transform a DataArray time x lat x lon into a time x region dataset

    v : xarray.DataArray (will be reindexed onto ds_mask)
    ds_mask : xarray.Dataset of binary masks (the regions)
    """
    assert v.dims == ("time", "lat", "lon") # no need to be more general here

    v_extract = v.reindex(lon=ds_mask.lon.values, lat=ds_mask.lat.values)

    subregions = list(ds_mask)

    return xa.DataArray(
        np.array([_regional_average(v_extract.values, ds_mask[k].values) for k in subregions]).T,
        name=name, dims=("time", "region"), coords={"time": v.time, "region": subregions})


def calc_regional_averages(v, ds_mask, name=None):
    """Transform a DataArray time x lat x lon into a time x region dataset

    v : xarray.DataArray (will be reindexed onto ds_mask)
    ds_mask : xarray.Dataset of binary masks (the regions)
    """
    # silence some warnings that may occur
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')

        return _calc_regional_averages_unfiltered(v, ds_mask, name=name)


def open_map_files(indicator, simus):
    files = [indicator.get_path(**simu) for simu in simus]
    return open_mfdataset(files, combine='nested', concat_dim="time")[indicator.ncvar]

def get_all_subregion(region, weights="latWeight"):
    with open_region_mask(region, weights) as mask:
        return list(mask)

def _open_regional_data_from_csv(indicator, simu, regions, weights="latWeight", admin=True, **kwargs):
    """This function loads data from the CSV files
    """
    if regions is None:
        regions = get_all_regions()
    files = [get_regional_averages_file(indicator.name, simu["climate_forcing"], simu["climate_scenario"],
                                    region, weights, impact_model=simu.get("model"), **kwargs)
                                    for region in regions]

    n0 = len(files)

    missing_regions = [region for (f, region) in zip(files, regions) if not f.exists()]
    files = [f for f in files if f.exists()]
    if len(files) == 0:
        raise FileNotFoundError(f"No regional files found for {indicator.name} {simu['climate_forcing']} {simu['climate_scenario']} {simu.get('model')}")

    if len(files) < n0:
        logger.info(f"Missing regions {missing_regions}")
        logger.warning(f"Only {len(files)} out of {n0} files exist. Skip the missing ones.")

    if admin:
        dfs = pd.concat([pd.read_csv(file, index_col=0) for file in files], axis=1)  # concat region and their admin boundaries
    else:
        dfs = pd.concat([pd.read_csv(file, index_col=0).iloc[:, :1] for file in files], axis=1)  # only use the first column (full region)

    # make sure we have dates as index (and not just years, cause the calling function needs dates)
    dfs.index = pd.to_datetime(dfs.index.astype(str))

    return xa.DataArray(dfs,
        coords=[dfs.index, dfs.columns],
        dims=["time", "region"],
        name=indicator.ncvar,
        )


def _open_regional_data(indicator, simu, regions=None, weights="latWeight",
                        admin=True, save=True, load=True, load_csv=False, all_masks=None):
    """Load the gridded netCDF and compute the regional averages on the fly
    """
    file = indicator.get_path(**simu)
    file_regional = indicator.get_path(**simu, regional=True, regional_weight=weights)

    if load and file_regional.exists():
        logger.info(f"Load regional averages from {file_regional}")
        return open_dataset(file_regional)[indicator.ncvar]

    elif load_csv:
        logger.info(f"Load regional averages from CSV files")
        ds = _open_regional_data_from_csv(indicator, simu, regions, weights, admin)
        # if save:
        #     logger.info(f"Write regional averages to {file_regional}")
        #     ds.to_netcdf(file_regional, encoding={indicator.ncvar: {'zlib': True}})
        return ds

    if regions is None:
        regions = get_all_regions()

    if all_masks is None:
        all_masks = get_merged_masks(regions, weights, admin)

    with open_dataset(file) as ds:
        region_averages = calc_regional_averages(ds[indicator.ncvar], all_masks, name=indicator.ncvar)

    if save:
        logger.info(f"Write regional averages to {file_regional}")
        file_regional.parent.mkdir(parents=True, exist_ok=True)
        region_averages.to_netcdf(file_regional, encoding={indicator.ncvar: {'zlib': True}})

    return region_averages


def open_regional_files(indicator, simus, **kwargs):
    return xa.concat([_open_regional_data(indicator, simu, **kwargs)
                      for simu in simus], dim="time") # historical and future


def open_files(indicator, simus, regional=False, **kwargs):
    if regional:
        return open_regional_files(indicator, simus, **kwargs)
    else:
        return open_map_files(indicator, simus)



def main():

    ALL_MASKS = sorted([o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*")])

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=CONFIG["indicator"], help="includes additional, secondary indicator with specific monthly statistics")
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    group.add_argument("--region", nargs='+', default=ALL_MASKS, choices=ALL_MASKS)
    group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])

    o = parser.parse_args()
    setup_logger(o)

    masks = preload_masks(o.region, o.weights)

    for variable in o.variable+o.indicator:
        indicator = Indicator.from_config(variable)
        simus = [simu for simu in indicator.simulations
                 if _matches(simu["climate_forcing"], o.model)
                 and _matches(simu["climate_scenario"], o.experiment)
                 and _matches(simu.get("model"), o.impact_model)]

        for model, group_ in groupby(sorted(simus, key=lambda x: x["climate_forcing"]), key=lambda x: x["climate_forcing"]):
            for experiment, group__ in groupby(sorted(group_, key=lambda x: x["climate_scenario"]), key=lambda x: x["climate_scenario"]):
                for impact_model, group in groupby(sorted(group__, key=lambda x: x.get("model")), key=lambda x: x.get("model")):
                    group = list(group)
                    logger.info(f"{variable}, {model}, {experiment}:: {len(group)} simulations")

                    todo = [(region, weights) for region in o.region for weights in o.weights if o.overwrite or not get_regional_averages_file(variable, model, experiment, region, weights, impact_model=impact_model).exists()]

                    if not todo:
                        logger.info(f"{variable}, {model}, {experiment} region-mask averages already exist")
                        continue

                    elif len(todo) < len(o.region)*len(o.weights):
                        logger.info(f"{variable}, {model}, {experiment}:: {len(todo)} / {len(o.region)*len(o.weights)} region-mask left to process")

                    else:
                        logger.info(f"{variable}, {model}, {experiment}:: process {len(todo)} region-mask")

                    results = {}
                    tempfiles = set()

                    for file in tqdm.tqdm([indicator.get_path(**simu) for simu in group]):
                    # for file in tqdm.tqdm(get_files(variable, model, experiment, frequency=o.frequency, simulation_round=o.simulation_round)):

                        with open_dataset(file) as ds:

                            v = ds[indicator.ncvar].load()

                            for region, weights in todo:

                                # save intermediary file in the ISIMIP tree structure (useful if time-slices are added later, e.g. historical back in time)
                                tag = f"region-{region}-{weights}"
                                # filetmp = Path(str(file).replace("global", tag))
                                filetmp = Path(str(file)+f".{tag}")
                                tempfiles.add(filetmp)

                                if filetmp.exists():
                                # if filetmp.exists() and not o.overwrite: # uncomment if the region itself if redefined
                                    res = xa.open_dataset(filetmp)[variable].load()

                                else:
                                    if (region, weights) not in masks:
                                        continue

                                    res = calc_regional_averages(v, masks[(region, weights)], name=variable)

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
                        ofile = get_regional_averages_file(variable, model, experiment, region, weights, impact_model=impact_model)

                        logger.info(f"Write region average output {ofile}")
                        ofile.parent.mkdir(exist_ok=True, parents=True)
                        # result.to_dataset(name=o.variable).to_netcdf(ofile)
                        result.to_pandas().to_csv(ofile)

                    for file in tempfiles:
                        if file.exists():
                            logger.debug(f"Remove {file}")
                            file.unlink()

if __name__ == "__main__":
    main()