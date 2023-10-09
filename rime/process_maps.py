# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Apr  6 11:36:24 2023

@author: werning, byers

To Do:
    
    - HANDLE SSPs THAT DON'T EXIST FOR IMPACT DATA'

"""

import dask
import glob
import numpy as np
import os
import pyam
import time
import xarray as xr
import yaml
from dask import delayed
from dask.distributed import Client
from rime.process_config import *
from rime.rime_functions import *


# from dask.diagnostics import Profiler, ResourceProfiles, CacheProfiler
dask.config.set(scheduler="processes")
dask.config.set(num_workers=num_workers)
client = Client()

# dask.config.set(scheduler='synchronous')


with open(yaml_path, "r") as f:
    params = yaml.full_load(f)


# %% Load scenario data

df_scens_in = pyam.IamDataFrame(fname_input_scenarios)
dft = df_scens_in.filter(variable=temp_variable)
dft = np.round(dft.as_pandas()[pyam.IAMC_IDX + ["year", "value", "Ssp_family"]], 2)
# Replace & fill missing SSP scenario allocation
# dft.Ssp_family.replace(sspdic, inplace=True) # metadata must have Ssp_faily column. If not SSP2 automatically chosen
dft.loc[dft.Ssp_family.isnull(), ssp_meta_col] = "ssp2"

dft = pyam.IamDataFrame(dft)

# % interpolate maps


# %% Test multiple scenarios, 1 indicator

print("Test multiple scenarios, 1 indicator")
start = time.time()

ind = "precip"
var = "sdii"
short = params["indicators"][ind][var]["short_name"]

ssp = "ssp2"
# Test multiple scenarios, 1 indicator
files = glob.glob(os.path.join(impact_data_dir, ind, f"*{short}_{ssp}*{ftype}.nc4"))
mapdata = xr.open_mfdataset(
    files, preprocess=remove_ssp_from_ds, combine="nested", concat_dim="gmt"
)

# df = pyam.IamDataFrame(dft)

map_out_MS = map_transform_gmt_multi_dask(
    dft.filter(model="POLES ADVANCE"),
    mapdata,
    years,
    use_dask=True,
    gmt_name="threshold",
    interpolation=interpolation,
)

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in map_out_MS.data_vars}
filename = f"{output_folder_maps}scenario_maps_multiscenario_{ftype}.nc"
map_out_MS.to_netcdf(filename, encoding=encoding)


print("FINISHED Test multiple scenarios, 1 indicator")
print(f"{time.time()-start}")

# 5 scenarios, 1 indi = 15 seconds
# 5 scenarios, 1 indi = 13s   dask=False
# 5 scenarios, 1 indi = 6s    dask=True
# 47 scenarios, 1 indi = 128s, = 2.7 s/s
# 60 scenarios, 1 indi = 120s, 2 s/s, 24 workers

# 880 scenarios, 1 indi = 2109s = 2.5 s/s
# 880 scenarios, 1 indi = 1686s = 1.9 s/s, 36 workers


# %% Test 1 scenario, multiple indicators Faster with non dask

print("Test 1 scenario, multiple indicators")
start = time.time()
ssp = "ssp2"
mapdata = xr.Dataset()
indicators = [
    "cdd",
    "precip",
    "dri",
    "dri_qtot",
    "ia_var",
    "ia_var_qtot",
    "sdd_18p3",
    "sdd_24p0",
]  #'heatwave']

for ind in indicators:
    print(ind)
    for var in params["indicators"][ind]:
        short = params["indicators"][ind][var]["short_name"]
        files = glob.glob(
            os.path.join(impact_data_dir, ind, f"*{short}_{ssp}*{ftype}.nc4")
        )
        mapdata[short] = xr.open_mfdataset(
            files, preprocess=remove_ssp_from_ds, combine="nested", concat_dim="gmt"
        )[short]

map_out_MI = map_transform_gmt_multi_dask(
    dft.filter(model="AIM*", scenario="SSP1-34"),
    mapdata,
    years,
    use_dask=False,
    gmt_name="threshold",
)

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in map_out_MI.data_vars}
filename = f"{output_folder_maps}scenario_maps_multiindicator_{ftype}.nc"
map_out_MI.to_netcdf(filename, encoding=encoding)

print("FINISHED 1 scenario, multiple indicators")
print(f"{time.time()-start}")

# 1 scenarios, 5 indis = 14 seconds
# 1 scenario, 8 indis = 22s, = 1.8s/s dask=False
# 1 scenario, 12 indis = 62s, = 5s/s dask=False, 24 workers.
# 1...                   32s     dask=True
