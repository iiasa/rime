# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:35:40 2022

@author: byers
"""
# from alive_progress import alive_bar
import glob

# import numpy as np
import os
import pyam
import re
import xarray as xr
import time

# try:

# ab_present = True
# except:
#     print("alive_progress not installed")
#     ab_present = False

# start = time.time()

from rime.rime_functions import *
from rime.process_config import *

few_vars = False

# %% Settings config

years = list(range(yr_start, yr_end + 1))


# %% List of indicators and files

# Define a list of table data files to read in
os.chdir(wdtable_input)
files = glob.glob(table_output_format.replace("|", "*"))


# indicator_subset = ['cdd','dri','ia_var_qtot',]#
# indicator_subset = ['seas_qtot','tr20','wsi'] #'heatwave','precip','sdd_24p0',
# indicator_subset = ['sdd_20p0']
# indicator_subset = ['heatwave']
# if indicator_subset:
# files = [str for str in files if any(sub in str for sub in indicator_subset)]

# files = files[0:3]
# files = files[1:2]
# files = files[4:5] # heatwav & iavar
# files = files[5:10]
# files = files[10:12]
# files = files[12:15]
files = files[15:]


indicators = []

for x in files:
    indicators.append(re.split(table_output_format, x)[1])

assert len(indicators) == len(files)


# %% loop through indicator files


for i, ind in enumerate(zip(indicators, files)):
    print(ind)
    istart = time.time()
    dfin = pyam.IamDataFrame(f"{wdtable_input}{ind[1]}")

    if ind[0] == "heatwave":
        hws = ["hw_95_10*", "hw_99_5*"]
        crop = dfin.filter(variable=hws).variable
        #     crop = [str for str in crop if any(sub in str for sub in ['High','Low'])==False]
        dfin.filter(variable=crop, inplace=True)

    elif region == "COUNTRIES":
        stems = [x.split("|")[0] for x in dfin.variable]
        subs1 = []
        subs = [
            "|Exposure|Land area",
            "|Exposure|Population",
            "|Exposure|Population|%",
            "|Hazard|Absolute|Land area weighted",
            "|Hazard|Absolute|Population weighted",
            "|Hazard|Risk score|Population weighted",
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

    df[["SSP", "GMT"]] = df.scenario.str.split("_", expand=True)
    df["GMT"] = df["GMT"].str.replace("p", ".").astype("float")

    # df.drop(columns=['model', 'scenario'], inplace=True)

    # if few_vars:
    #     df = df.loc[df.variable.str.contains('High')]

    small_vars = list(set([x.split("|")[0] for x in dfin.variable]))
    # if ab_present:
    # with alive_bar(total=len(small_vars),
    #                title='Processing', length=10, force_tty=True,
    #                bar='circles',
    #                spinner='elements') as bar:

    #         print('alive bar present')
    # Apply function here
    for vari in small_vars:
        df_ind = loop_interpolate_gmt(
            df.loc[df.variable.str.startswith(vari)], yr_start, yr_end
        )
        # dfbig = pd.concat([dfbig, df_ind])
        print(f"dfbig: indicator {ind[0]}: {time.time()-istart}")

        # % Convert and save out to xarray
        # dfbig.dropna(how='all')

        dfp = df_ind.melt(
            id_vars=[
                "model",
                "scenario",
                "variable",
                "region",
                "unit",
                "SSP",
                "GMT",
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
                .set_index(["gmt", "year", "ssp", "region"])
                .to_xarray()
            )
            # dx.attrs['unit'] = dx.assign_coords({'unit':dx.unit.values[0,0,0,0]})
            dsout[indicator] = dx["value"].to_dataset(name=indicator)[indicator]
            dsout[indicator].attrs["unit"] = dx.unit.values[0, 0, 0, 0]
            # dsout = dsout[indicator].assign_coords({'unit':dx.unit.values[0,0,0,0]})

        dsout["ssp"] = [x.upper() for x in dsout["ssp"].values]
        # dsout = dsout.drop_vars('unit')

        # % Write out
        print("Writing out... ")
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in dsout.data_vars}
        filename = f"{output_dir}{vari}_{region}.nc"
        dsout.to_netcdf(filename, encoding=encoding)
        # if ab_present:
        # bar()

    # =============================================================================
