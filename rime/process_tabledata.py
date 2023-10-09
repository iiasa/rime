# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 00:24:36 2022

@author: byers
"""
if __name__ == "__main__":
    from process_config import *

    #  from alive_progress import alive_bar
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.distributed import Client
    import glob
    import numpy as np
    import pandas as pd
    import pyam
    import re
    import time
    import xarray as xr
    import dask
    from rime_functions import *

    # from pandas import InvalidIndexError
    # dask.config.set(scheduler='threads')  # overwrite default with multiprocessing scheduler

    filesall = glob.glob(fname_input_climate)

    # files = filesall
    # files = filesall[:2]
    # files = filesall[2:6]
    # files = filesall[7:9] # problem in 6?
    # files = filesall[9:12]
    # files = filesall[12:15]
    files = filesall[15:]

    if len(files) == 0:
        print("no files!")
        dsfsdfsdfsdf

    # load input IAMC scenarios file
    df_scens_in = pyam.IamDataFrame(fname_input_scenarios)

    if parallel:
        dask.config.set(
            scheduler="processes"
        )  # overwrite default with multiprocessing scheduler
        dask.config.set(num_workers=num_workers)
        print(f'Number of Dask workers: {dask.config.get("num_workers")}')
        if env == "pc":
            dask.config.set({"temporary-directory": "C:\\Temp"})
        else:
            dask.config.set({"temporary-directory": f"D:\\{user}"})
        client = Client()

    # %% New function parallel

    # % # Load aggregated file, and test with temperature pathway
    # =============================================================================

    # %%

    for year_res in year_resols:
        for f in files:
            start = time.time()

            # Get variable name
            v1 = f.split(f"_{region}")[0]
            v2 = v1.split("\\")[-1]

            years = range(2015, 2101, year_res)

            # load input climate impacts data file
            ds = xr.open_mfdataset(f)
            ds = ds.sel(year=years)

            # Filter for temperature variable
            dft = df_scens_in.filter(variable=temp_variable)

            if few_scenarios:
                dft = dft.filter(scenario="*SSP*")
                dft = dft.filter(Category=["C1", "C2", "C3", "C8"])
                dft = dft.filter(Category=["C1*"])
                # dft = dft.filter(scenario='R2p1_SSP2-PkBudg900', keep=False)
                if very_few_scenarios:
                    dft = dft.filter(model="WIT*", scenario="*")

            #  assign SSPs
            dft = dft.filter(year=years)
            dft = np.round(
                dft.as_pandas()[pyam.IAMC_IDX + ["year", "value", "Ssp_family"]], 3
            )
            sspdic = {1.0: "SSP1", 2.0: "SSP2", 3.0: "SSP3", 4.0: "SSP4", 5.0: "SSP5"}
            dft.Ssp_family.replace(
                sspdic, inplace=True
            )  # metadata must have Ssp_faily column. If not SSP2 automatically chosen
            dft.loc[dft.Ssp_family.isnull(), ssp_meta_col] = "SSP2"
            dfX = pyam.IamDataFrame(dft)

            # % Full thing

            varis = list(ds.data_vars.keys())[:lvaris]

            # Subset to fewer indicators
            # varis = list(ds.data_vars.keys())
            # varis = [str for str in list(ds.data_vars.keys()) if any(sub in str for sub in ['High','Low'])==False]
            # lvaris = len(varis)
            # dsi = ds[list(ds.data_vars.keys())[:x]]

            dsi = ds[varis]
            print(f"real len variables = {len(varis)}")

            dft = dfX.timeseries().reset_index()

            # Fix duplicate temperatures
            dft = dft.apply(fix_dupes, axis=1)
            dft.reset_index(inplace=True, drop=True)

            # Convert to dask array
            # meta_df = pd.DataFrame(columns=dft.columns, dtype=float)#, name='meta')
            # dic = {k:str for k in ['model',
            #  'scenario',
            #  'region',
            #  'variable',
            #  'unit',
            #  'ssp_family',]}

            # meta_df = meta_df.astype(dic)

            if parallel:
                ddf = dd.from_pandas(dft, npartitions=1000)

                # dfx = dft.iloc[0].squeeze()  # FOR DEBUIGGING THE FUNCTION
                outd = ddf.apply(
                    calculate_impacts_gmt, dsi=dsi, axis=1, meta=("result", None)
                )
                # outdd = client.map(ddf.apply(calculate_impacts_gmt, dsi=dsi,  axis=1))

                with ProgressBar():
                    # try:
                    df_new = outd.compute(
                        num_workers=num_workers
                    )  # scheduler='processes')
                    print(f" Applied:  {time.time()-start}")
                # except(InvalidIndexError):
                # print(f'PROBLEM {f}')
            else:
                df_new = dft.apply(
                    calculate_impacts_gmt, dsi=dsi, axis=1
                )  # .compute(scheduler='processes')

            expandedd = pd.concat([df_new[x] for x in df_new.index])
            print(f" Done:  {time.time()-start}")

            filename = f"{wd}{wd2}{output_folder_tables}{input_scenarios_name}_rcre_output_{region}_{v2}_{year_res}yr{test}.csv"

            expandedd.to_csv(filename, encoding="utf-8", index=False)
            print(f" Saved: {region} yrs={year_res}\n  {time.time()-start}")
            print(f"{len(dsi.data_vars)} variables")

            # keep_cols = [x for x in expandedd.columns if type(x)==str]
            # keep_cols = keep_cols + list(range(2015,2101,5))
            # out_small = expandedd[keep_cols]
            # filename_small = f'{wd}{wd2}{output_folder}{input_scenarios_name}_rcre_output_10yr_{region}.csv'
            # out_small.to_csv(filename_small, encoding='utf-8')
            # print(f' Saved:  {time.time()-start}')


# %%
