# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Apr  6 11:36:24 2023

@author: werning, byers
Execute this script ideally in GitHub/rime folder

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

# from dask.distributed import Client
from rimeX.legacy.process_config import *
from rimeX.legacy.rime_functions import *
from rimeX.legacy.utils import *


# from dask.diagnostics import Profiler, ResourceProfiles, CacheProfiler
dask.config.set(scheduler="processes")
dask.config.set(num_workers=num_workers)
# client = Client()

# dask.config.set(scheduler='synchronous')


with open("rime" + yaml_path, "r") as f:
    params = yaml.full_load(f)


# %% Load scenario data

df_scens_in = pyam.IamDataFrame(fname_input_scenarios)
dft = df_scens_in.filter(variable=temp_variable)
dft = ssp_helper(dft)


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
    files, preprocess=remove_ssp_from_ds, combine="nested", concat_dim="gwl"
)

mapdata = tidy_mapdata(mapdata)


df = dft.filter(model="POLES GE*")

map_out_MS = map_transform_gwl_wrapper(
    df,
    mapdata,
    years,
    use_dask=True,
    gwl_name="gwl",
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
# year=2055
mapdata = xr.Dataset()
indicators = [
    "cdd",
    "precip",
    # "dri",
    # "dri_qtot",
    # "iavar",
    # "ia_var_qtot",
    "sdd_18p3",
    # "sdd_24p0",
    "wsi",
]  #'heatwave']

for ind in indicators:
    print(ind)
    for var in params["indicators"][ind]:
        short = params["indicators"][ind][var]["short_name"]
        files = glob.glob(
            os.path.join(impact_data_dir, ind, f"*{short}_{ssp}*{ftype}.nc4")
        )
        mapdata[short] = xr.open_mfdataset(
            files, preprocess=remove_ssp_from_ds, combine="nested", concat_dim="gwl"
        )[short]

mapdata = tidy_mapdata(mapdata)


map_out_MI = map_transform_gwl_wrapper(
    dft.filter(model="AIM*", scenario="SSP1-34"),
    mapdata,
    years=years,
    use_dask=False,
    gwl_name="gwl",
)

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in map_out_MI.data_vars}
filename = f"{output_folder_maps}scenario_maps_multiindicator_{ftype}.nc"
dask.config.set(scheduler="single-threaded")
map_out_MI.to_netcdf(filename, encoding=encoding)

print("FINISHED 1 scenario, multiple indicators")
print(f"{time.time()-start}")

# 1 scenarios, 5 indis = 14 seconds
# 1 scenario, 8 indis = 22s, = 1.8s/s dask=False
# 1 scenario, 12 indis = 62s, = 5s/s dask=False, 24 workers.
# 1...                   32s     dask=True

#%% Test plot dashboard

indicators = ["cdd", "dri"]

filename = "test_map.html"
plot_maps_dashboard(
    map_out_MI,
    indicators=indicators,
    filename=filename,
    year=2055,
    cmap="magma_r",
    shared_axes=True,
    clim=None,
)
os.startfile(filename)

#%%

filename = "test_map.html"
plot_maps_dashboard(
    map_out_MI,
    filename=filename,
    year=2055,
    cmap="magma_r",
    shared_axes=True,
    clim=None,
)
os.startfile(filename)
