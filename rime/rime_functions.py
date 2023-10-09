# -*- coding: utf-8 -*-
"""
rime_functions.py
Set of core functions for RIME
"""

import dask
from dask import delayed
import numpy as np
import pandas as pd
import pyam
import xarray as xr


def loop_interpolate_gmt(df, yr_start, yr_end):
    """
    Loop through the variables and regions in df and interpolate the impacts data
    between the global warming levels
    Parameters
    ----------
    df : pandas.DataFrame()
        df is in IAMC wide format, with columns model/scenario/region/unit/variable/[years]
    yr_start : int  # 2010  # 2015
        start year of the interpolated data
    yr_end : int  # 2100  # 2099
        end year of the interpolated data

    Returns
    -------
    df_ind : pandas.DataFrame()

    """
    # if type(df_ind)==None:
    df_ind = pd.DataFrame(columns=df.columns)
    regions = df.region.unique()
    SSPs = df.SSP.unique()

    for variable in df.variable.unique():
        print(variable)
        for ssp in SSPs:
            for region in regions:
                dfs = df.loc[
                    (df.region == region) & (df.SSP == ssp) & (df.variable == variable)
                ]
                if len(dfs) > 0:
                    dfs.index = range(0, 50 * len(dfs), 50)
                    dfs = dfs.reindex(index=range(50 * len(dfs)))
                    tcols = pyam.IAMC_IDX + ["SSP"]
                    dfs[tcols] = dfs.loc[0, tcols]

                    cols = list(
                        range(yr_start, yr_end + 1),
                    ) + ["GMT"]

                    dfs[cols] = dfs[cols].interpolate()
                    dfs.drop_duplicates(inplace=True)
                    df_ind = pd.concat([df_ind, dfs])

    return df_ind


# =============================================================================
# Table outputs
# =============================================================================


def calculate_impacts_gmt(
    dfx, dsi, prefix_indicator="RIME|", ssp_meta_col="ssp_family", gmt_below=1.2
):
    """
    Takes in a set of scenarios of GMT and dataset of climate impacts by GWL,
    and returns a table of climate impacts per scenario through time.

    Parameters
    ----------
    dfx : dask.DataFrame
        dask.DataFrame in the format of an IAMC table (see pyam.IamDataFrame) with scenario(s) as rows and only 1 variable for the global mean temperature timeseries. Needs to have a column specifying the SSP to be assumed.
    dsi : xarray.Dataset
        dimensions ['gmt','year','ssp','region'] and each variable is a climate indicator
    prefix_indicator : str, optional
        DESCRIPTION. The default is 'RIME|'. Use this to change the output indicator prefix in the table data variable column.
    ssp_meta_col : str, optional
        DESCRIPTION. The default is 'Ssp_family'. Use this to change the name of the meta column used to assin the SSP per scenario
    gmt_below : float, optional
        DESCRIPTION. The default is 1.5. Assign all gmt values below gmt_below to the value of gmt_below (in case not enough data)

    Returns
    -------
    idf : pyam.IamDataFrame
        pyam.IamDataFrame with variables for the impact indicators for each scenario(s).

    """

    model = dfx["model"]
    scenario = dfx["scenario"]

    ssp = dfx[ssp_meta_col.lower()]

    if ssp not in ["SSP1", "SSP2", "SSP3"]:
        ssp = "SSP2"

    # tt.loc[tt.value<gmt_below, 'value'] = gmt_below

    years = [x for x in dfx.index if type(x) == int]
    tt = dfx[years]

    idf = pd.DataFrame(columns=pyam.IAMC_IDX + years)

    for indicator in list(dsi.data_vars):
        # print(indicator)

        edf = pd.DataFrame(
            columns=pyam.IAMC_IDX
        )  # dont have years because otherwise joins with empty rows

        edf["region"] = dsi.region.values
        edf["model"] = model
        edf["scenario"] = scenario
        edf["variable"] = f"{prefix_indicator}{indicator}"
        # try:
        edf["unit"] = dsi[indicator].attrs["unit"]
        # except(InvalidIndexError):
        #     edf['unit'] = dsi[indicator].unit
        dsd = dsi.sel(ssp=ssp)[indicator]

        tgt_y = xr.DataArray(years, dims="points")
        tgt_g = xr.DataArray(tt[years].values, dims="points")
        try:
            agh = dsd.sel(year=tgt_y, gmt=tgt_g, method="nearest").to_dataframe(
                name=indicator
            )
            agh = agh.reset_index()
            aghp = agh.pivot_table(index="region", columns="year", values=indicator)
            edf = edf.join(aghp, on="region")

        except Exception as e:
            print(f"skip {model} {scenario}")

        idf = pd.concat([idf, edf]).reset_index(drop=True)

    # print('end func')
    return idf


def fix_dupes(dfti, years):
    """
    Function that modifies GMT temperatures minutely, in case there are duplicates in the series.

    Parameters
    ----------
    dfti : pandas.DataFrame,
        a file from which to take transform and latitude objects

    Returns
    -------
    """

    vs = dfti[years]
    ld = len(dfti[years])
    lu = len(set(dfti[years]))

    if ld != lu:
        # print('')
        seen = set()
        vsn = []
        for x in vs:
            if x not in vsn:
                vsn.append(x)
            else:
                vsn.append(x + np.random.uniform(-0.005, 0.005))
        dfti[years] = vsn
    return dfti


# =============================================================================
# Mpas
# =============================================================================


def map_transform_gmt(
    df1,
    mapdata,
    var_name,
    years,
    map_array=xr.Dataset(),
    caution_checks=True,
    drawdown_max=0.15,
    gmt_name="threshold",
    interpolation=0.01,
    temp_min=1.2,
    temp_max=3.5,
):
    """
        Takes in a set of scenarios of GMT and dataset of climate impacts by GWL,
        and returns a table of climate impacts per scenario through time.

        Parameters
        ----------
        df1 : dask.DataFrame
            dask.DataFrame in the format of an IAMC table (see pyam.IamDataFrame) with scenario(s) as rows and only 1 variable for the global mean temperature timeseries. Needs to have a column specifying the SSP to be assumed.
        mapdata : xarray.Dataset
            dimensions ['gmt','year','ssp','region'] and each variable is a climate indicator
            .....
    ....
    ...

        Returns
        -------
        idf : pyam.IamDataFrame
            pyam.IamDataFrame with variables for the impact indicators for each scenario(s).

    """

    if len(df1.variable) > 1:
        raise Exception("Error: more than 1 variable in DataFrame")
    if len(df1.meta) > 1:
        raise Exception("Error: more than 1 model-scenario in DataFrame")

    # Should only be 1 scenario
    for model, scenario in df1.index:
        # Get the GMT values from df
        gmt_values = df1.filter(year=years).data["value"].values
        gmt_values = np.round(gmt_values, 2)

        if caution_checks:
            # check if gmt at end reduction post peak greater than 0.2C
            drawdown = np.round(gmt_values.max() - gmt_values[-1], 2)
            if drawdown > drawdown_max:
                print(f"Warning! Overshoot  drawdown: {drawdown}. Scenario SKIPPED")
                continue
            if gmt_values.max() > temp_max:
                print(
                    f"Warning! Max temperature above {temp_max}°C  {gmt_values.max()}, data thereafter not possible"
                )
                continue
            if gmt_values.min() < temp_min:
                print(
                    f"Warning! Min temperature below {temp_min}°C  {gmt_values.min()}, data before not possible"
                )

        # Replace the values outside range with 999 index (of nans)
        gmt_values[(gmt_values < temp_min) | (gmt_values > temp_max)] = 999

        # Load and prepare the spatial impact data to be transformed
        # Rename 3rd dimension to 'gmt'
        try:
            mapdata = mapdata.rename_vars({gmt_name: "gmt"})
        except ValueError:
            print("")

        # Get indicator name
        short = list(mapdata.data_vars)[0]
        # provide new interpolated vector (e.g. 0.01 gmt resolution)
        new_thresholds = np.arange(
            mapdata.gmt.min(), mapdata.gmt.max() + interpolation, interpolation
        )
        new_thresholds = [round(i, 2) for i in new_thresholds]
        data_interp = mapdata.interp(gmt=new_thresholds)

        # Add dummy data for out of range values to last slice 999
        # last_slice = data_interp.isel(gmt=-1)
        new_slice = xr.DataArray([999], dims=("gmt",), coords={"gmt": [999]})
        data_interp = xr.concat([data_interp[short], new_slice], dim="gmt")
        data_interp["gmt"] = data_interp["gmt"].assign_coords(
            gmt=data_interp["gmt"].values
        )
        data_interp = data_interp.to_dataset()

        # Create new empty dataarray for the new data.
        map_array = map_array.assign(
            variables={
                var_name: (
                    ("lat", "lon", "year"),
                    np.full(
                        [len(data_interp.lat), len(data_interp.lon), len(years)], np.nan
                    ),
                )
            }
        )
        data_interp[short][
            -1,
            :,
            :,
        ] = np.full([len(data_interp.lat), len(data_interp.lon)], np.nan)

        # Create an array to store the updated data
        updated_data = data_interp[short].sel(gmt=gmt_values)

        # Update map_array
        map_array[var_name][:, :, : len(years)] = updated_data.transpose(
            "lat", "lon", "gmt"
        )

        # Drop other coords
        dc = [x for x in map_array.coords if x not in ["lon", "lat", "year"]]
        if len(dc) > 0:
            map_array.drop(dc)
    return map_array


def map_transform_gmt_multi_dask(
    df,
    mapdata,
    years,
    gmt_name="threshold",
    use_dask=True,
    temp_min=1.2,
    temp_max=3.5,
    drawdown_max=0.15,
    interpolation=0.01,
):
    """
    Takes in a set of scenarios of GMT and dataset of climate impacts by GWL,
    and returns a table of climate impacts per scenario through time.

    Parameters
    ----------
    df : dask.DataFrame
        dask.DataFrame in the format of an IAMC table (see pyam.IamDataFrame) with scenario(s) as rows and only 1 variable for the global mean temperature timeseries. Needs to have a column specifying the SSP to be assumed.
    mapdata : xarray.Dataset
        dimensions ['gmt','year','ssp','region'] and each variable is a climate indicator
    filename : str, optional
        DESCRIPTION. The default is NONE. output filename as .csv (utf-8), otherwise returns idf
    # prefix_indicator : str, optional
    #     DESCRIPTION. The default is 'RIME|'. Use this to change the output indicator prefix.
    # ssp_meta_col : str, optional
    #     DESCRIPTION. The default is 'Ssp_family'. Use this to change the name of the meta column used to assin the SSP per scenario
    # gmt_below : float, optional
    #     DESCRIPTION. The default is 1.5. Assign all gmt values below gmt_below to the value of gmt_below (in case not enough data)

    Returns
    -------
    idf : pyam.IamDataFrame
        pyam.IamDataFrame with variables for the impact indicators for each scenario(s).

    """

    map_array = xr.Dataset()

    if len(df.index) > 1:
        if len(mapdata.data_vars) == 1:
            # =============================================================================
            #             1 indicator, multi-scenario mode
            # =============================================================================
            print("Single indicator mode")
            delayed_tasks = []

            for model, scenario in df.index:
                modelstrip = model
                repdic = [" ", "/", ",", "."]
                for char in repdic:
                    modelstrip = modelstrip.replace(char, "_")
                var_name = f"{modelstrip}_{scenario}"  # f'{model}_{scenario}'
                print(var_name)

                df1 = df.filter(model=model, scenario=scenario)

                if use_dask:
                    # Create delayed task for map_transform_gmt
                    delayed_map_transform = delayed(map_transform_gmt)(
                        df1, mapdata, var_name, years, map_array
                    )
                    delayed_tasks.append(delayed_map_transform)
                else:
                    map_array = map_transform_gmt(
                        df1, mapdata, var_name, years, map_array
                    )
            if use_dask:
                # Compute delayed tasks concurrently
                computed_results = dask.compute(*delayed_tasks)

                # Merge the computed results into map_array
                for result in computed_results:
                    map_array.update(result)

        else:
            print("Error! Multiple IAM scenarios and spatial indicators detected.")
            raise ValueError(
                "Make sure that one of df or mapdata is only 1 scenario/indicator"
            )

    else:
        # =============================================================================
        #             1 scenario, multi-indicator mode
        # =============================================================================
        print("Single scenario mode")
        delayed_tasks = []

        # Iterate through spatial indicators in DataSet
        for var_name in mapdata.data_vars:
            print(var_name)
            df1 = df
            if use_dask:
                # Create delayed task for map_transform_gmt_dask
                delayed_map_transform = delayed(map_transform_gmt)(
                    df1, mapdata, var_name, years, map_array
                )
                delayed_tasks.append(delayed_map_transform)
            else:
                map_array = map_transform_gmt(df1, mapdata, var_name, years, map_array)
        if use_dask:
            # Compute delayed tasks concurrently
            computed_results = dask.compute(*delayed_tasks)

            # Merge the computed results into map_array
            for result in computed_results:
                map_array.update(result)

    map_array = map_array.assign_coords(
        coords={"lon": map_array.lon, "lat": map_array.lat, "year": years}
    )

    return map_array


def co2togmt_simple(cum_CO2, regr=None):
    """
    Takes in vector of cumulative CO2 values and calculates Global mean surface
    air temperature (p50).
    Parameters
    ----------
    cum_CO2 : int, float, np.array, pandas.Series
        Value of cumulative CO2, from 2020 to net-zero CO2 emissions or end of century, in Gt CO2.
    regr : dict, optional
        'slope' and 'intercept' values for line. The default is None, in which
        case parameters from AR6 assessment are used. Provide {'slope': m,
        'intercept': x} to define own linear relationship, otherwise 'AR5'.

    Returns
    -------
    Global mean surface air temperature (p50).

    """
    import pandas as pd
    import numpy as np

    if regr == None:
        # use default AR6 paramters
        slope = 0.0005099869587542405
        intercept = 1.3024249460191835
    elif type(regr) == dict:
        # User provided parameters
        slope = regr["slope"]
        intercept = regr["intercept"]
    elif type(regr) == str:
        if regr == "AR6":
            # use default AR6 paramters
            slope = 0.0005099869587542405
            intercept = 1.3024249460191835
        elif regr == "AR5":
            # use default AR6 paramters
            slope = 0.0043534
            intercept = 1.36555
        else:
            print("Warning: specification not recognized")
            raise ("Error: specification not recognized")
    elif type(regr) == pd.DataFrame:
        from scipy.stats import linregress

        # use the first two columns
        x = regr.columns[0]
        y = regr.columns[1]
        slope, intercept, r, p, se = linregress(regr[x], regr[y])
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"r2: {r}")

    gmt = slope * cum_CO2 + intercept

    return gmt


def preprocess(ds):
    var = list(ds.keys())[0]
    short = f'{var.rsplit("_ssp")[0]}'
    return ds.rename({list(ds.keys())[0]: short})
