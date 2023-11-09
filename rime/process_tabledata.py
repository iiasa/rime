# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 00:24:36 2022

@author: byers
"""
if __name__ == "__main__":
    from process_config import *
    from rime_functions import *

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

    # from pandas import InvalidIndexError
    # dask.config.set(scheduler='threads')  # overwrite default with multiprocessing scheduler

    filesall = glob.glob(fname_input_climate)

    # files = filesall
    files = filesall[:2]
    # files = filesall[2:6]
    # files = filesall[7:9] # problem in 6?
    # files = filesall[9:12]
    # files = filesall[12:15]
    # files = filesall[15:]

    if len(files) == 0:
        raise Exception("No files!")

    # load input IAMC scenarios file
    df_scens_in = pyam.IamDataFrame(fname_input_scenarios)

    mode = "CO2"
    if mode == "CO2":
        print(
            "CO2 mode: Global mean temperatures will be derived from response \
              to cumulative CO2 emissions."
        )
    elif mode == "GMT":
        print("GMT mode: Global mean temperatures provided as input.")

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

    # %% Normal testing with temperature variable

    for year_res in year_resols:
        for f in files:
            start = time.time()

            # Get variable name for the filename output
            v1 = f.split(f"_{region}")[0]
            v2 = v1.split("\\")[-1]

            years = range(2015, 2101, year_res)

            #############################
            # CLIMATE DATA PRE-PROCESSING
            # load input climate impacts data file
            ds = xr.open_mfdataset(f)
            ds = ds.sel(year=years)

            # % Full thing
            varis = list(ds.data_vars.keys())[:lvaris]
            # Subset to fewer indicators
            # varis = list(ds.data_vars.keys())
            # varis = [str for str in list(ds.data_vars.keys()) if any(sub in str for sub in ['High','Low'])==False]
            # lvaris = len(varis)
            # dsi = ds[list(ds.data_vars.keys())[:x]]
            dsi = ds[varis]
            print(f"# of variables = {len(varis)}")

            ##############################
            # SCENARIO DATA PRE-PROCESSING
            # Filter for temperature variable

            if mode == "GMT":
                dfp = df_scens_in.filter(variable=temp_variable)
            elif mode == "CO2":
                dfp = prepare_cumCO2(df_scens_in, years=years, use_dask=True)
                ts = dfp.timeseries().apply(co2togmt_simple)
                ts = pyam.IamDataFrame(ts)
                ts.rename(
                    {
                        "variable": {ts.variable[0]: "RIME|GSAT_tcre"},
                        "unit": {ts.unit[0]: "°C"},
                    },
                    inplace=True,
                )
                # Export data to check error and relationships
                # ts.append(dfp).to_csv('c://users//byers//downloads//tcre_gmt_output.csv')
                dfp = ts
                dfp.meta = df_scens_in.meta.copy()
            dfp = dfp.filter(year=years)

            if few_scenarios:
                dfp = dfp.filter(scenario="*SSP*")
                dfp = dfp.filter(Category=["C1", "C2", "C3", "C8"])
                dfp = dfp.filter(Category=["C1*"])
                # dfp = dfp.filter(scenario='R2p1_SSP2-PkBudg900', keep=False)
                if very_few_scenarios:
                    dfp = dfp.filter(model="REMIND 2.1*", scenario="*")  # (4)

            #  assign SSPs
            dfp = ssp_helper(dfp, ssp_meta_col="Ssp_family", default_ssp="SSP2")

            # Fix duplicate temperatures
            dft = dfp.timeseries().reset_index()
            dft = dft.apply(fix_duplicate_temps, years=years, axis=1)
            dft.reset_index(inplace=True, drop=True)

            ###########################
            # START PROCESSING

            if parallel:
                """
                For parallel processing, convert dft as a wide IAMC pd.Dataframe
                into a dask.DataFrame.
                """
                ddf = dd.from_pandas(dft, npartitions=1000)

                # dfx = dft.iloc[0].squeeze()  # FOR DEBUIGGING THE FUNCTION
                outd = ddf.apply(
                    calculate_impacts_gmt, dsi=dsi, axis=1, meta=("result", None)
                )
                # outdd = client.map(ddf.apply(calculate_impacts_gmt, dsi=dsi,  axis=1))

                with ProgressBar():
                    # try:
                    df_new = outd.compute(num_workers=num_workers)
                    print(f" Applied:  {time.time()-start}")
                # except(InvalidIndexError):
                # print(f'PROBLEM {f}')
            else:
                df_new = dft.apply(calculate_impacts_gmt, dsi=dsi, axis=1)

            expandedd = pd.concat([df_new[x] for x in df_new.index])
            print(f" Done:  {time.time()-start}")

            filename = f"{wd}{wd2}{output_folder_tables}{input_scenarios_name}_RIME_output_{region}_{v2}_{year_res}yr{test}.csv"

            # expandedd.to_csv(filename, encoding="utf-8", index=False)
            print(f" Saved: {region} yrs={year_res}\n  {time.time()-start}")
            print(f"{len(dsi.data_vars)} variables, {len(dfp.meta)} scenarios")
