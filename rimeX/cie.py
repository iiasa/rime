import pandas as pd
import numpy as np
import xarray as xa
from scipy.io import loadmat

from rimeX.datasets.manager import get_datapath
from rimeX.config import CONFIG
from rimeX.preproc.quantilemaps import make_quantilemap_prediction, get_filepath
from rimeX.preproc.digitize import get_binned_isimip_file
from rimeX.records import make_models_equiprobable
from rimeX.emulator import recombine_gmt_ensemble, recombine_gmt_vectorized, load_magicc_ensemble

def load_magicc_mat(magicc_scenario):
    magicc_path = get_datapath(f"MAGICC6-RCPs/MAGICC_{magicc_scenario}_SURFACE_TEMP_GLOBAL.mat")
    magicc_data = loadmat(magicc_path)
    gmt = pd.DataFrame(magicc_data['DATA'], index=magicc_data['TIME'].flatten())

    y1, y2 = CONFIG["emulator.projection_baseline"]
    offset = CONFIG["emulator.projection_baseline_offset"]
    gmt = (gmt - gmt.loc[y1:y2].mean() + offset).loc[1980:2100]
    return gmt


def predict_from_records(gmt, indicator_name, region, subregion, season, weight, quantiles=[0.5, .05, .95], vectorized=False, samples=5000, clip=True, seed=42, equiprobable_models=True):

    # load the impact data (old CSV form)
    fp = get_binned_isimip_file(indicator_name, region, subregion, weight, season)
    impact_data = pd.read_csv(fp)
    impact_data_records = impact_data.to_dict("records")

    if equiprobable_models:
        make_models_equiprobable(impact_data_records)

    if vectorized:
        if clip:
            gmt = gmt.clip(lower=impact_data["warming_level"].min(), upper=impact_data["warming_level"].max())

        result_df = recombine_gmt_vectorized(impact_data_records, gmt, samples=samples, seed=seed).quantile(quantiles, axis=1).T

    else:
        # count samples and calculate quantiles
        result_df = recombine_gmt_ensemble(impact_data_records, gmt, quantiles)

    return result_df


def predict_from_quantilemap(gmt, indicator_name, region, subregion, season, weight, quantiles=[0.5, .05, .95], samples=5000, clip=True, seed=42, suffix="_eq", **kw):

    fp = get_filepath(indicator_name, season=season, suffix=suffix, region=region, regional_weights=weight)

    with xa.open_dataset(fp) as ds:
        impact_data = ds[indicator_name].sel(region=subregion).load()

    return make_quantilemap_prediction(impact_data, gmt, samples=samples, quantiles=quantiles, clip=clip, seed=seed, **kw).T.to_pandas()
