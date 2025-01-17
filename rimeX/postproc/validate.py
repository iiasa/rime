import requests
import numpy as np
from scipy.stats import lognorm, norm
import pandas as pd
import xarray as xa
import io
import re

from rimeX.logs import logger
from rimeX.datasets import get_datapath
# from rimeX.compat import FastIamDataFrame
from rimeX.compat import _simplify
from rimeX.emulator import recombine_gmt_ensemble
from rimeX.stats import fit_dist_minimize
from rimeX.preproc.global_average_tas import load_global_average_tas
from rimeX.preproc.digitize import get_binned_isimip_file
from rimeX.preproc.digitize import get_indicator_units, transform_indicator, CONFIG

CONFIG["isimip.simulation_round"] = ["ISIMIP3b"]
# CONFIG["preprocessing.projection_baseline"] = y1, y2 = baseline = (1986, 2005)
CONFIG["preprocessing.projection_baseline"] = y1, y2 = baseline = (1995, 2014)

vlabels = {
    "tas": f"tas (°C)\nwarming since {y1}-{y2}",
    "pr": f"pr (%)\nchange since {y1}-{y2}",
    "rx5day": f"Wettest 5-Day Period (%)\nchange since {y1}-{y2}",
    }

slabels = {s: f"RCP {s[-2]}.{s[-1]}" for s in ["rcp26", "rcp45", "rcp60", "rcp85"]}


def load_impact_file(variable, country):
    # impact_file = f"/mnt/PROVIDE/climate_impact_explorer/isimip2/isimip_binned_data/{variable}/{country}/{country}/latWeight/{variable}_{country.lower()}_{country.lower()}_annual_latweight_bytime_21yrs_natvarFalse_clim_models-equi.csv"
    # /mnt/PROVIDE/climate_impact_explorer/isimip3/isimip_binned_data/pr/RUS/RUS/latWeight/pr_rus_rus_annual_latweight_21-yrs_baseline-1986-2005.csv
    # /mnt/PROVIDE/climate_impact_explorer/isimip2/isimip_binned_data/pr/RUS/RUS/latWeight/pr_rus_rus_annual_latweight_21-yrs_baseline-1986-2005.csv
    # impact_file = f"/mnt/PROVIDE/climate_impact_explorer/isimip2/isimip_binned_data/{variable}/{country}/{country}/latWeight/{variable}_{country.lower()}_{country.lower()}_annual_latweight_21-yrs_baseline-1986-2005.csv"
    # impact_file = f"/mnt/PROVIDE/climate_impact_explorer/isimip2/running-21-years/isimip_binned_data/{variable}/{country}/{country}/latWeight/{variable}_{country.lower()}_{country.lower()}_annual_latweight_21-yrs_baseline-1986-2005.csv"
    impact_file = get_binned_isimip_file(variable, country, country, "latWeight", "annual", backend="csv")
    impact_data_records = pd.read_csv(impact_file).to_dict('records')
    return impact_data_records

# rng = np.random.default_rng(42)
# N = 5000

def load_global_average_tas2(variable, model, scenario):
    ts = load_global_average_tas(variable, model, scenario)
    if ts is None:
        raise ValueError(f"Could not load {variable} for {model} {scenario}")
    ts.index = pd.to_datetime(ts.index).year
    annual = ts.groupby(ts.index).mean()
    return annual

def load_resampled_esm_gmt(scenario):
    tss = []
    models = []
    for model in CONFIG["isimip.ISIMIP3b.models"]:
        tsh = load_global_average_tas2("tas", model, "historical")
        try:
            ts = load_global_average_tas2("tas", model, scenario)
        except ValueError:
            logger.warning(f"Could not load GMT for {model} {scenario}")
            continue
        ts = ts.reindex(np.arange(2015, 2101), method="ffill")
        ts = pd.concat([tsh, ts], axis=0)
        tss.append(ts)
        models.append(model)
    gmt = pd.concat(tss, axis=1)
    gmt.columns = models
    y1, y2 = CONFIG["emulator.projection_baseline"]
    offset = CONFIG["emulator.projection_baseline_offset"]
    return gmt - gmt.loc[y1:y2].mean() + offset
    # i = rng.integers(0, len(gmt), size=N)
    # return gmt.iloc[:, i]

def run_rimeX_cie(variable, scenario, country, upper=.95, lower=.05):
    # magicc_file = f"/mnt/PROVIDE/climate_impact_explorer/artificial_MAGICC_output/{scenario}_MAGICC.csv"
    # gmt_ensemble = pd.read_csv(magicc_file, sep="\s+", index_col=0)
    impact_data_records = load_impact_file(variable, country)
    gmt_ensemble = load_resampled_esm_gmt(scenario)
    gmt_ensemble = gmt_ensemble.reindex(np.arange(2015, 2101, 5))
    quantiles = [.5, upper, lower]
    return recombine_gmt_ensemble(impact_data_records, gmt_ensemble, quantiles)


def parse_filename(filename):
    pattern = r"(?P<model>.*?)_(?P<scenario>.*?)_(?P<variable>.*)_(?P<country>.*?)_(?P<weights>.*?)\.csv"
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    else:
        raise ValueError("Filename does not match the expected pattern")

def load_isimip_timeseries(variable, scenario, country):
    from pathlib import Path
    model = "*"
    matches = sorted([str(f) for f in Path(f"/mnt/PROVIDE/climate_impact_explorer/isimip3/isimip_regional_data/{country}/latWeight").glob(f"{model}_{scenario}_{variable}*")])
    parsed_data = [parse_filename(str(Path(f).name)) for f in matches]
    models = [d["model"] for d in parsed_data]
    series = [pd.read_csv(f, index_col=0)[country].loc[:"2120"] for f in matches]
    df = pd.DataFrame(series).T
    df.columns = models
    if df.index.dtype == "O":
        df.index = pd.to_datetime(df.index)
        df = df.groupby(df.index.year).mean()
    return df