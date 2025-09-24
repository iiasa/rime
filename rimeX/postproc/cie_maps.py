import os
import glob
import re
import numpy as np
import xarray as xr
from rimeX.config import CONFIG

# === helper functions ===

def adjust_map_values(var, impact_data):
    if var == "total_annual_precipitation":
        impact_data = impact_data * 365
    elif var in ["moderate_drought",
                 "severe_drought",
                 "extreme_drought",
                 "superextreme_drought",
                 "maximumextreme_drought"]:
        impact_data = impact_data * 100
    elif var in ['fwils', 'fwixd']:
        impact_data = impact_data.astype(float)
        impact_data = impact_data * 1.15741 * 10 ** -14
    elif var in ['tas', 'tasmax', 'tasmin']:
        impact_data = impact_data - 273.15

    if var in ['pr', 'total_annual_precipitation', 'prsn']:
        impact_data = impact_data * 60 * 60 * 24

    if var in ['ps', 'psl']:
        impact_data = impact_data / 100

    return impact_data


def same_sign(a: xr.DataArray, b: xr.DataArray, zero_strict: bool = False) -> xr.DataArray:
    if zero_strict:
        result = ((a > 0) & (b > 0)) | ((a < 0) & (b < 0)) | ((a == 0) & (b == 0))
    else:
        result = ((a >= 0) & (b >= 0)) | ((a < 0) & (b < 0))
    return result.astype("int8")


def get_model_lists(indicator: str, data_path: str):
    """
    Collect model/IM and model/IM/scenario combinations for a given indicator.
    Supports both NetCDF and CSV patterns:
      (nc1) {model}_{scenario}_{indicator}{...}.nc
      (nc2) {model}_{impact_model}_{scenario}_{indicator}{...}.nc
      (csv1) {model}_{scenario}_{indicator}_regional_{aggregation}_{temporal_scale}_{start}_{end}.csv
      (csv2) {model}_{impact_model}_{scenario}_{indicator}_regional_{aggregation}_{temporal_scale}_{start}_{end}.csv
    Only parses files directly under {data_path}/indicators/{indicator}/{scenario}/{model}/
    """
    indicator_dir = os.path.join(data_path, "indicators", indicator)
    if not os.path.exists(indicator_dir):
        return [], []

    gcm_im = set()
    gcm_im_scenario = set()

    for scenario in os.listdir(indicator_dir):
        scenario_dir = os.path.join(indicator_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        for model in os.listdir(scenario_dir):
            model_dir = os.path.join(scenario_dir, model)
            if not os.path.isdir(model_dir):
                continue

            for fname in os.listdir(model_dir):
                if not (fname.endswith(".nc") or fname.endswith(".csv")):
                    continue

                name = os.path.splitext(fname)[0]  # strip extension
                if indicator not in name:
                    continue

                idx = name.find(indicator)
                prefix = name[:idx].rstrip("_")
                parts = prefix.split("_")

                if len(parts) == 2:
                    # pattern (1): model, scenario
                    model_name, scenario_name = parts
                    gcm_im.add(model_name)
                    gcm_im_scenario.add(f"{model_name}_{scenario_name}")

                elif len(parts) == 3:
                    # pattern (2): model, impact_model, scenario
                    model_name, impact_model, scenario_name = parts
                    gcm_im.add(f"{model_name}_{impact_model}")
                    gcm_im_scenario.add(f"{model_name}_{impact_model}_{scenario_name}")

                else:
                    continue  # ignore unexpected

    return sorted(gcm_im), sorted(gcm_im_scenario)



# === settings ===
ROOT_DIR = "/mnt/PROVIDE/climate_impact_explorer"
DATA_DIR = f"{ROOT_DIR}/isimip3/running-21-years/quantilemaps"
OUTPUT_DIR = "/mnt/PROVIDE/climate_impact_explorer/map_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

overwrite = True
indicators_to_process = ['TXx', 'cooling_degree_days', 'hurs', 'pr', 'rice_yield', 'runoff', 'sfcwind', 'tasmax', 'wheat_yield',
'annual_drought_intensity','daily_temperature_variability', 'huss', 'prsn', 'river_discharge', 'tasmin',  'consecutive_dry_days', 'extreme_daily_rainfall', 'maize_yield', 
'ps', 'rlds', 'rx1day', 'soy_yield', 'wet_bulb_temperature',  'consecutive_tropical_nights', 'heavy_precipitation_days', 'number_of_wet_days', 'rsds', 'rx5day', 'tas']          # e.g. ["tas", "pr"]; leave empty for all

# === loop indicators / files ===
indicator_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "*")))

for indicator_path in indicator_dirs:
    indicator = os.path.basename(indicator_path)
    OUTPUT_DIR_indicator = f'{OUTPUT_DIR}/{indicator}'
    os.makedirs(OUTPUT_DIR_indicator, exist_ok=True)

    if indicators_to_process and indicator not in indicators_to_process:
        continue

    suffix = "_quantilemaps_wl13-qb11-eq.nc"
    pattern = os.path.join(indicator_path, f"{indicator}_*{suffix}")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)

        # parse temporal_average robustly
        temporal_average = None
        if fname.endswith(suffix):
            core = fname[:-len(suffix)]
            parts = core.rsplit("_", 1)
            if len(parts) == 2:
                temporal_average = parts[1]
        if temporal_average is None:
            raise ValueError(f"Cannot parse temporal_average from {fname}")

        print(f"Processing {fname} (indicator={indicator}, temporal_average={temporal_average})")

        ds = xr.load_dataset(fpath)

        # extract quantile DataArrays
        indicator_data_05 = adjust_map_values(indicator, ds[indicator].sel(quantile=0.5))
        indicator_data_08 = adjust_map_values(indicator, ds[indicator].sel(quantile=0.8))
        indicator_data_02 = adjust_map_values(indicator, ds[indicator].sel(quantile=0.2))
    
        # fetch model lists
        gcm_im, gcm_im_scenario = get_model_lists(indicator, os.path.dirname(os.path.dirname(os.path.dirname(DATA_DIR))))
        print(gcm_im)
        print(gcm_im_scenario)

        warming_levels = np.round(np.arange(1.0, 3.9, 0.1), 1)

        for wl1 in warming_levels:
            
            # === NEW: abs case ===
            abs_values = indicator_data_05.interp(warming_level=wl1)
            model_agreement = same_sign(
                    indicator_data_08.interp(warming_level=wl1),
                    indicator_data_02.interp(warming_level=wl1)
                )
            model_agreement_rel_1 = same_sign(
                    indicator_data_08.interp(warming_level=wl1)-indicator_data_08.interp(warming_level=1.0),
                    indicator_data_02.interp(warming_level=wl1)- indicator_data_02.interp(warming_level=1.0)
                )
            ds_abs = xr.Dataset({"var": abs_values, "model_agreement": model_agreement, "model_agreement_rel_1.0": model_agreement_rel_1})
            ds_abs = ds_abs.assign_coords(warming_level=float(wl1))
            ds_abs.attrs["indicator"] = indicator
            ds_abs.attrs["temporal_average"] = temporal_average
            ds_abs.attrs["GCM-IM input"] = gcm_im
            ds_abs.attrs["GCM-IM-scenario available"] = gcm_im_scenario

            out_name_abs = f"{indicator}_{wl1:.1f}_abs_{temporal_average}.nc"
            out_path_abs = os.path.join(OUTPUT_DIR_indicator, out_name_abs)

            if (not overwrite) and os.path.exists(out_path_abs):
                print(f" -> Skipping existing {out_name_abs}")
            else:
                ds_abs.to_netcdf(out_path_abs)
                print(f" -> Saved {out_name_abs}")

            # === differences case ===
            for wl2 in warming_levels[warming_levels < wl1]:
                values = indicator_data_05.interp(warming_level=wl1) - indicator_data_05.interp(warming_level=wl2)
                model_agreement = same_sign(
                    indicator_data_08.interp(warming_level=wl1) - indicator_data_08.interp(warming_level=wl2),
                    indicator_data_02.interp(warming_level=wl1) - indicator_data_02.interp(warming_level=wl2)
                )

                ds_out = xr.Dataset({"var": values, "model_agreement": model_agreement})
                ds_out = ds_out.assign_coords(warming_level=float(wl1), second_warming_level=float(wl2))
                ds_out.attrs["indicator"] = indicator
                ds_out.attrs["temporal_average"] = temporal_average
                ds_out.attrs["GCM-IM input"] = gcm_im
                ds_out.attrs["GCM-IM-scenario available"] = gcm_im_scenario

                out_name = f"{indicator}_{wl1:.1f}_{wl2:.1f}_{temporal_average}.nc"
                out_path = os.path.join(OUTPUT_DIR_indicator, out_name)

                if (not overwrite) and os.path.exists(out_path):
                    print(f" -> Skipping existing {out_name}")
                    continue

                ds_out.to_netcdf(out_path)
                print(f" -> Saved {out_name}")
