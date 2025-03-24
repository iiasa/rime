# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:35:40 2022

@author: byers
"""
from alive_progress import alive_bar
import glob

# import numpy as np
import os
import pyam
import re
import xarray as xr
import time

try:

    ab_present = True
except:
    print("alive_progress not installed")
    ab_present = False

# start = time.time()

from rime.rime_functions import *
from rime.process_config import *
from rime.utils import *

few_vars = False

# %% Settings config

years = list(range(yr_start, yr_end + 1))


# %% List of indicators and files

# Define a list of table data files to read in
os.chdir(wdtable_input)
files = glob.glob(table_output_format.replace("|", "*"))

indicator_subset = ['cdd', 'dri', 'dri_qtot', 'iavar', 'iavar_qtot',
                    'hw_95_3', 'hw_95_5', 'hw_95_7', 'hw_95_10',
                    'hw_97_3', 'hw_97_5', 'hw_97_7', 'hw_97_10',
                    'hw_99_3', 'hw_99_5', 'hw_99_7', 'hw_99_10',
                    'hwd_95_3', 'hwd_95_5', 'hwd_95_7', 'hwd_95_10',
                    'hwd_97_3', 'hwd_97_5', 'hwd_97_7', 'hwd_97_10',
                    'hwd_99_3', 'hwd_99_5', 'hwd_99_7', 'hwd_99_10',
                    'pr_r10', 'pr_r20', 'pr_r95p', 'pr_r99p', 'sdii',
                    'seas', 'seas_qtot', 'sdd_c', 'sdd_c_24p0', 'sdd_c_20p0',
                    'sdd_c_18p3', 'tr20', 'wsi']

files = [str for str in files if any(sub in str for sub in indicator_subset)]

indicators = []

for x in files:
    indicators.append(re.split(table_output_format, x)[1])

assert len(indicators) == len(files)


# %% loop through indicator files


for i, ind in enumerate(zip(indicators, files)):
    print(ind)
    istart = time.time()
    dfin = pyam.IamDataFrame(f"{wdtable_input}{ind[1]}")
    dfin = dfin.filter(model='Climate Solutions', variable='*|50.0th Percentile')
    

    if ind[0] == "heatwave":
        hws = ["hw_95_10*", "hw_99_5*"]
        crop = dfin.filter(variable=hws).variable
        #     crop = [str for str in crop if any(sub in str for sub in ['High','Low'])==False]
        dfin.filter(variable=crop, inplace=True)

    elif region == "COUNTRIES":
        stems = [x.split("|")[0] for x in dfin.variable]
        subs1 = []
        subs = [
            "|Exposure|Land area|50.0th Percentile",
            "|Exposure|Population|50.0th Percentile",
            "|Exposure|Population|%|50.0th Percentile",
            "|Hazard|Absolute|Land area weighted|50.0th Percentile",
            "|Hazard|Absolute|Population weighted|50.0th Percentile",
            "|Hazard|Risk score|Population weighted|50.0th Percentile",
        ]
        for x in list(set(stems)):
            for i in subs:
                subs1.append(f"{x}{i}")
        dfin.filter(variable=subs1, inplace=True)

    # if i==0:
    #    df = pd.read_csv(ind[1])
    #    df = pyam.IamDataFrame(ind[1]).timeseries().reset_index()
    #    # df.columns = [x.lower() if type(x)==str elif x[0]=== for x in df.columns]
    #    dfbig = pd.DataFrame(columns=df.columns)

    df = dfin.interpolate(
        time=years,
    )
    df = df.timeseries().reset_index()
    df.dropna(how="all", inplace=True)

    df[["SSP", "GWL"]] = df.scenario.str.split("_", expand=True)
    df["GWL"] = df["GWL"].str.replace("p", ".").astype("float")
    df["variable"] = df["variable"].str.replace("|50.0th Percentile", "")

    # df.drop(columns=['model', 'scenario'], inplace=True)

    # if few_vars:
    #     df = df.loc[df.variable.str.contains('High')]

    small_vars = list(set([x.split("|")[0] for x in dfin.variable]))
    if ab_present:
        with alive_bar(
            total=len(small_vars),
            title="Processing",
            length=10,
            force_tty=True,
            bar="circles",
            spinner="elements",
        ) as bar:

            print("alive bar present")
        # Apply function here
        for vari in small_vars:
            df_ind = loop_interpolate_gwl(
                df.loc[df.variable.str.startswith(vari)], yr_start, yr_end
            )
            # dfbig = pd.concat([dfbig, df_ind])
            print(f"dfbig: indicator {ind[0]}: {time.time()-istart}")

            # % Convert and save out to xarray (todo - make function)
            # dfbig.dropna(how='all')

            dfp = df_ind.melt(
                id_vars=[
                    "model",
                    "scenario",
                    "variable",
                    "region",
                    "unit",
                    "SSP",
                    "GWL",
                ],
                value_vars=years,
                var_name="year",
            )  # change to df_big if concatenating multiple

            dfp.columns = [x.lower() for x in dfp.columns]

            dsout = xr.Dataset()

            for indicator in dfp.variable.unique():
                print(indicator)
                dx = (
                    dfp.loc[dfp.variable == indicator]
                    .set_index(["gwl", "year", "ssp", "region"])
                    .to_xarray()
                )
                # dx.attrs['unit'] = dx.assign_coords({'unit':dx.unit.values[0,0,0,0]})
                dsout[indicator] = dx["value"].to_dataset(name=indicator)[indicator]
                dsout[indicator].attrs["unit"] = dx.unit.values[0, 0, 0, 0]
                # dsout = dsout[indicator].assign_coords({'unit':dx.unit.values[0,0,0,0]})

            dsout["ssp"] = [x.upper() for x in dsout["ssp"].values]
            # dsout = dsout.drop_vars('unit')

            # Ne

            # % Write out
            print("Writing out... ")
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in dsout.data_vars}
            filename = f"{output_dir}{vari}_{region}.nc"
            dsout.to_netcdf(filename, encoding=encoding)
            if ab_present:
                bar()

    # =============================================================================
