# -*- coding: utf-8 -*-
"""
rime_functions.py
Set of core functions for RIME
"""

import dask
import dask.dataframe as dd
from dask import delayed
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import math
import numpy as np
import pandas as pd
import pyam
from scipy.stats import linregress
import xarray as xr

from rime.utils import check_ds_dims


def loop_interpolate_gwl(df, yr_start, yr_end, interval=50, gmt_out_resolution=0.01, gmt_smooth_resolution=0.05):
    """
    Loop through the variables and regions in df and interpolate the impacts data
    between the global warming levels (typically at 0.5 °C).
    Parameters
    ----------
    df : pandas.DataFrame()
        df is in IAMC wide format, with columns model/scenario/region/unit/variable/[years]
    yr_start : int  # 2010  # 2015
        start year of the interpolated data
    yr_end : int  # 2100  # 2099
        end year of the interpolated data
    interval : int, default = 50
        interval between global warming levels for interpolation. e.g. interval=50 and
        providing data with impacts data at 0.5 °C resolutions, would interpolate to
        0.01 deg C resolution.

    Returns
    -------
    df_ind : pandas.DataFrame()

    """
    years = list(range(yr_start, yr_end))
    df_ind = pd.DataFrame(columns=df.columns)
    regions = df.region.unique()
    SSPs = df.SSP.unique()
    
    def round_down(x, a):
        return math.floor(x / a) * a
    def round_up(x, a):
        return math.ceil(x / a) * a        

    for variable in df.variable.unique():
        print(variable)
        for ssp in SSPs:
            for region in regions:
                dfs = df.loc[
                    (df.region == region) & (df.SSP == ssp) & (df.variable == variable)
                ].reset_index(drop=True)
                # if len(dfs) > 0:
                    # dfs.index = range(0, interval * len(dfs), interval)
                    # dfs = dfs.reindex(index=range(interval * len(dfs)))
                    # tcols = pyam.IAMC_IDX + ["SSP"]
                    # dfs[tcols] = dfs.loc[0, tcols]

                    # cols = list(
                        # range(yr_start, yr_end + 1),
                    # ) + ["GWL"]

                    # dfs[cols] = dfs[cols].interpolate()
                    # dfs.drop_duplicates(inplace=True)
                    # df_ind = pd.concat([df_ind, dfs])
                    
                ## new
                if len(dfs) > 0:                
                    min_gwl = np.round(round_down(dfs['GWL'].min(), gmt_smooth_resolution),2)
                    max_gwl = np.round(round_up(dfs['GWL'].max(), gmt_smooth_resolution),2)
                    new_gwl = np.arange(min_gwl, max_gwl + gmt_out_resolution, gmt_out_resolution).round(2)
                    new_df = pd.DataFrame({'GWL': new_gwl})
                    
                                       
                    tcols = pyam.IAMC_IDX + ["SSP"]
                    dfs[tcols] = dfs.loc[0, tcols]

                    cols = list(
                        range(yr_start, yr_end + 1),
                    ) + ["GWL"]
                    
                    
                    dfs[years] = dfs[years].interpolate(axis=1, limit_direction='both')
                    # dfs[years] = dfs[years].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
                    
                    merged_df = pd.merge(new_df, dfs, on='GWL', how='left')
                    merged_df[tcols] = dfs.loc[0, tcols]
                    
                    dfs = merged_df.interpolate(method='linear', limit_direction='both')
                    
                    df_ind = pd.concat([df_ind, dfs])

    if 'gmt' in df_ind.columns:
        df_ind.drop(columns='gmt')
    return df_ind


# =============================================================================
# Table outputs
# =============================================================================


def table_impacts_gwl(
    dfx,
    dsi,
    prefix_indicator="RIME|",
    ssp_meta_col="Ssp_family",
):
    """
    Takes in a table of scenarios of GWL and a dataset of climate impacts by GWL,
    and returns a table of climate impacts per scenario through time. The

    Parameters
    ----------
    dfx : dask.DataFrame
        dask.DataFrame in the format of an IAMC wide table (see pyam.IamDataFrame.timeseries()) with scenario(s) as rows and only 1 variable for the global mean temperature timeseries. Needs to have a column specifying the SSP to be assumed.
    dsi : xarray.Dataset
        dimensions ['gwl','year','ssp','region'] and each variable is a climate indicator. This format is an xarray file of impacts,
        prepared from the output of the function loop_interpolate_gwl.
    prefix_indicator : str, optional
        The default is 'RIME|'. Use this to change the output indicator prefix in the table data variable column.
    ssp_meta_col : str, optional
        The default is 'Ssp_family'. Use this to change the name of the meta column used to assin the SSP per scenario
    # gwl_below : float, optional (currently not usd)
    #     The default is 1.5. Assign all gwl values below gwl_below to the value of gwl_below (in case not enough data)

    Returns
    -------
    idf : pyam.IamDataFrame
        pyam.IamDataFrame with variables for the impact indicators for each scenario(s).

    """

    model = dfx["model"]
    scenario = dfx["scenario"]

    ssp = dfx[ssp_meta_col]  # <<< isue here?

    # Need to check SSP integration here.
    if ssp not in ["SSP1", "SSP2", "SSP3"]:
        ssp = "SSP2"

    # tt.loc[tt.value<gwl_below, 'value'] = gwl_below

    years = [x for x in dfx.index if type(x) == int]
    tt = dfx[years]

    # New dataframe in IAMC wide format for all output data.
    idf = pd.DataFrame(columns=pyam.IAMC_IDX + years)

    for indicator in list(dsi.dataset.data_vars):
        # print(indicator)

        # For each indicator, new df
        edf = pd.DataFrame(
            columns=pyam.IAMC_IDX
        )  # dont have years because otherwise joins with empty rows

        edf["region"] = dsi.dataset.region.values
        edf["model"] = model
        edf["scenario"] = scenario
        edf["variable"] = f"{prefix_indicator}{indicator}"
        # try:
        edf["unit"] = dsi.dataset[indicator].attrs["unit"]
        # except(InvalidIndexError):
        #     edf['unit'] = dsi[indicator].unit
        dsd = dsi.dataset.sel(ssp=ssp)[indicator]

        tgt_y = xr.DataArray(years, dims="points")
        tgt_g = xr.DataArray(tt[years].values, dims="points")

        # Do selection (lookup) of datapoints in dsd ix along the years & gwl points.
        # try:
        agh = dsd.sel(year=tgt_y, gwl=tgt_g, method="nearest").to_dataframe(
            name=indicator
        )
        agh = agh.reset_index()
        aghp = agh.pivot_table(index="region", columns="year", values=indicator)
        edf = edf.join(aghp, on="region")

        # except Exception as e:
            # print(f"skip {model} {scenario}")

        idf = pd.concat([idf, edf]).reset_index(drop=True)

    return idf


# =============================================================================
# Maps
# =============================================================================


def map_transform_gwl(
    df1,
    mapdata,
    years,
    map_array=xr.Dataset(),
    var_name=None,
    caution_checks=True,
    include_orig_gwl=False,
    gwl_name="gwl",
    drawdown_max=0.15,
    temp_min=1.2,
    temp_max=3.5,
    interpolation=0.01,
):
    """
    Takes in one scenario of GWL and dataset of climate impacts by GWL,
    and returns a table of climate impacts for that scenario through time. Can be used on its own, but
    typically is called from map_transform_gwl_wrapper

    Parameters
    ----------
    df1 : dask.DataFrame, pandas.Dataframe()
        dask or pandas DataFrame in the format of a wide IAMC table (see pyam.IamDataFrame.timeseries()) with scenario(s) as rows and only 1 variable for the global mean temperature timeseries. Needs to have a column specifying the SSP to be assumed.
    mapdata : xarray.Dataset
        dimensions ['gwl','lat','lon'] and each variable is a climate indicator, gwl refers to global mean temperature.
    years : list, iterator
        The years for which the climate impacts are needed. Provided as a list of int, range() or similar.
    var_name : str, optional, optional
        Varaible / indicator name in the impacts dataset. Possibly not needed.
    map_array : xarray.Dataset(), optional
        xarray Dataset, by default empty, but can be passed into here if needing to add data.
    caution_checks : boolean, optional
        Checks key aspects of the input temperature scenario that could lead to unreliable results.
        drawdown - checks that the difference between peak and end-of-timeseries is not greater than x - default is 0.15 (°C).
        temp_min - checks whether time points in the scenario have lower temperatures than this and prints warning.
        temp_max - checks whether time points in the scenario have higher temperatures than this and prints warning.
    include_orig_gwl : boolean, optional
        Include the gwl coordinates in the output dataset. Default is False. Can only be True if in mode 2.
    drawdown_max : float, optional
        Maximum permitted level of temperature drawdown permitted after peak. Default is 0.15 (°C).
    gwl_name : str, optional
        If the input mapdata dimension for global mean temperatures is different to 'gwl', rename. Needs more testing, might not work.
    interpolation : float, optional
        Increment in degrees C at which to interpolate trhe data. Default is 0.01 °C.
    temp_min : float, optional
        Temperature values below this will be ignored. Limited by the extent of the climate impacts input data. Default is 1.2 °C.
    temp_max : float, optional
        Temperature values above this will be ignored. Limited by the extent of the climate impacts input data. Default is 3.5 °C.

    Returns
    -------
    idf : xarray.Dataset
        xarray.Dataset with variables for the impact indicators for each scenario(s) through time

    """

    # if len(df1.variable) > 1:
    #     raise Exception("Error: more than 1 variable in DataFrame")
    if len(df1.meta) > 1:
        raise Exception("Error: more than 1 model-scenario in DataFrame")

    # Should only be 1 scenario
    # Currently only works for 1 climate indicator
    for model, scenario in df1.index:
        # Get the GWL values from df
        gwl_values = df1.filter(year=years).data["value"].values
        gwl_values = np.round(gwl_values, 2)

        if caution_checks:
            # check if gwl at end reduction post peak greater than 0.2C
            drawdown = np.round(gwl_values.max() - gwl_values[-1], 2)
            if drawdown > drawdown_max:
                print(f"Warning! Overshoot  drawdown: {drawdown}. Scenario SKIPPED")
                # continue
            if gwl_values.max() > temp_max:
                print(
                    f"Warning! Max temperature above {temp_max}°C  {gwl_values.max()}, data thereafter not possible"
                )
                # continue
            if gwl_values.min() < temp_min:
                print(
                    f"Warning! Min temperature below {temp_min}°C  {gwl_values.min()}, data before not possible"
                )

        if gwl_values.max() > mapdata.gwl.values.max():
            temp_max = mapdata.gwl.values.max()
            print(f'Warning! Provided scenario temperatures are higher than the available climate impacts GWLs. Setting temp_max to {temp_max}')
        if gwl_values.min() < mapdata.gwl.values.min():
            temp_min = mapdata.gwl.values.min()
            print(f'Warning! Provided scenario temperatures are lower than the available climate impacts GWLs. Setting temp_min to {temp_min}')
      
        # Replace the values outside range with 999 index (of nans)
        gwl_values[gwl_values < temp_min] = 999
        gwl_values[gwl_values > temp_max] = 999

        # Load and prepare the spatial impact data to be transformed

        # Get indicator name
        # short = list(mapdata.data_vars)[0]
        # provide new interpolated vector (e.g. 0.01 gwl resolution)
        new_thresholds = np.arange(
            mapdata.gwl.min(), mapdata.gwl.max() + interpolation, interpolation
        )
        new_thresholds = [round(i, 2) for i in new_thresholds]
        mapdata_interp = mapdata.interp(gwl=new_thresholds)

        # Add dummy data for out of range values to last slice 999
        # last_slice = data_interp.isel(gwl=-1)
        new_slice = xr.DataArray([999], dims=("gwl",), coords={"gwl": [999]})
        data_interp = xr.Dataset()
        # mapdata_interp = mapdata_interp.to_dataset()
        for var_name in mapdata_interp.data_vars:
            data_interp[var_name] = xr.concat(
                [mapdata_interp[var_name], new_slice], dim="gwl"
            )

        # data_interp["gwl"] = data_interp["gwl"].assign_coords(
        # gwl=data_interp["gwl"].values
        # )
        # data_interp = data_interp.to_dataset()

        # Here todo: loop through variables in data_interp

        for var_name in data_interp.data_vars:
            map_array = map_array.assign(
                variables={
                    var_name: (
                        ("lat", "lon", "year"),
                        np.full(
                            [len(data_interp.lat), len(data_interp.lon), len(years)],
                            np.nan,
                        ),
                    )
                }
            )

        # for var_name in map_array.data_vars:
        #     data_interp[var_name][
        #         -1,
        #         :,
        #         :,
        #     ] = np.full([len(data_interp.lat), len(data_interp.lon)], np.nan)

        # Create an array to store the updated data
        # print(gwl_values)
        updated_data = data_interp.sel(gwl=gwl_values)
        updated_data = updated_data.transpose("lat", "lon", "gwl")
        updated_data = updated_data.rename_dims({"gwl": "year"})
        updated_data = updated_data.rename({"gwl":"year"})
        # updated_data = updated_data.set_index({"lon": "lon", "lat": "lat", "year": "year"}).reset_coords()
        updated_data = updated_data.assign_coords(
            coords={"lon": updated_data.lon, "lat": updated_data.lat, "year": years}
        )

        map_array = updated_data

        # Identify if other coordinates have been carried through with the Dataset and drop to
        # avoid confusion
        dc = [x for x in map_array.coords if x not in ["lon", "lat", "year", "gwl"]]
        if len(dc) > 0:
            map_array = map_array.drop(dc)

        # The gwl coordinate is carried through, but when combining multiple scenarios with different
        # gwl trajectories, this causes error in constructing the xr.DataSet. Drop as default, but
        # can be kept if using for only 1 scenario.
        if (include_orig_gwl==False) & ("gwl" in map_array.coords):
            try:
                map_array = map_array.drop("gwl")
            except ValueError:
                print('')

    return map_array


def map_transform_gwl_wrapper(
    df,
    mapdata,
    years,
    caution_checks=True,
    use_dask=True,
    include_orig_gwl=False,
    gwl_name="gwl",
    drawdown_max=0.15,
    temp_min=1.2,
    temp_max=3.5,
    interpolation=0.01,
):
    """
    Wrapper function of map_transform_gwl that can be used to leverage Dask for parallel processing of scenarios.
    Modes:
         1. a set of IAM scenarios of GWL and one xarray.DataSet climate impact indicator by GWLs, and returns an xarray.DataSet of the climate indicator per scenario through time.
         2. one IAM scenario of GWL and a an xarray Dataset with one or multiple climate impact indicators by GWLs, to return an xarray.DataSet of climate indicators for the scenario through time.

    Parameters
    ----------
    df : pyam.IamDataFrame
        pyam.IamDataFrame holding the scenario(s) with only the temperature variable. df is converted into appropriate dask or pandas Dataframe depending on whether use_dask=True/False.
    mapdata : xarray.Dataset
        xarray.Dataset holding the climate impacts data, with dimensions lat, lon and gwl. If only one (impact) variable (mode 1), then it can handle multiple IAM scenarios.
        If multiple (impact) variables (mode 2), then it can handle only one IAM scenario.
    years : list, iterator
        The years for which the climate impacts are needed. Provided as a list of int, range() or similar.
    gwl_name : str, optional
        If the input mapdata dimension for global mean temperatures is different to 'gwl', rename. Needs more testing, might not work.
    use_dask : boolean, optional
        Whether to process in parallel using Dask. Default is True. Small numbers of scenarios / indicators may not be faster due to the overheads that result from starting workers.
    include_orig_gwl : boolean, optional
        Include the gwl coordinates in the output dataset. Default is False. Can only be True if in mode 2.
    temp_min : float, optional
        Temperature values below this will be ignored. Limited by the extent of the climate impacts input data. Default is 1.2 °C.
    temp_max : float, optional
        Temperature values above this will be ignored. Limited by the extent of the climate impacts input data. Default is 3.5 °C.
    drawdown_max : float, optional
            Maximum permitted level of temperature drawdown permitted after peak. Default is 0.15 (°C).
    interpolation : float, optional
        Increment in degrees C at which to interpolate trhe data. Default is 0.01 °C.

    Returns
    -------
    map_array : xarray.DataSet
        Output xarray.Dataset which, depending on the mode:
        1. Multiple climate impact indicators (as variables), through time, for the given IAM scenario.
        2. One climate impact indicator, through time, for the given IAM scenarios (as variables).

    """

    if len(df.index) > 1:
        if len(mapdata.xrdataset.dims) > 2:
            map_array = mapdata.xrdataset.isel({gwl_name: 0}).reset_coords(drop=True)
            map_array = map_array.drop_vars(map_array.data_vars)
        else:
            map_array = xr.full_like(mapdata.xrdataset, np.nan).drop_vars(mapdata.xrdataset.data_vars)

        if len(mapdata.xrdataset.data_vars) == 1:
            # =============================================================================
            #   Mode 1:     1 indicator, multi-scenario mode
            # =============================================================================
            print("Single indicator mode (multi-scenarios possible)")
            delayed_tasks = []
            indicator = list(mapdata.xrdataset.data_vars)[0]

            for model, scenario in df.index:
                modelstrip = model
                repdic = [" ", "/", ",", "."]
                for char in repdic:
                    modelstrip = modelstrip.replace(char, "_")
                var_name = f"{modelstrip}_{scenario}"  # f'{model}_{scenario}'
                print(var_name)

                df1 = df.df.filter(model=model, scenario=scenario)

                if use_dask:
                    # Create delayed task for map_transform_gwl
                    delayed_map_transform = delayed(map_transform_gwl)(
                        df1,
                        mapdata.xrdataset,
                        years,
                        map_array,
                        var_name=None,
                        caution_checks=caution_checks,
                        include_orig_gwl=include_orig_gwl,
                        gwl_name=gwl_name,
                        drawdown_max=drawdown_max,
                        temp_min=temp_min,
                        temp_max=temp_max,
                        interpolation=interpolation,                    
                    )
                    # delayed_map_transform = delayed_map_transform.drop('gwl')
                    delayed_tasks.append(delayed_map_transform)
            
                # use_dask=False
                else:                
                    map_array[var_name] = map_transform_gwl(
                        df1, 
                        mapdata.xrdataset, 
                        years,
                        map_array, 
                        var_name,
                        caution_checks=caution_checks,
                        include_orig_gwl=include_orig_gwl,
                        gwl_name=gwl_name,
                        drawdown_max=drawdown_max,
                        temp_min=temp_min,
                        temp_max=temp_max,
                        interpolation=interpolation,
                    )[
                        indicator
                    ]  # .drop('gwl')  # drop here for alignment of coords (gwls are all different)

            # run this after the loop
            if use_dask:
                # Compute delayed tasks concurrently
                computed_results = dask.compute(*delayed_tasks)

                # Merge the computed results into map_array
                for result in computed_results:
                    map_array.update(result)

            
            map_array.attrs["indicator"] = indicator

        else:
            print("Error! Multiple IAM scenarios and spatial indicators detected.")
            raise ValueError(
                "Make sure that one of df or mapdata is only 1 scenario/indicator"
            )

    else:
        # =============================================================================
        #   Mode 2:      1 scenario, multi-indicator mode
        # =============================================================================
        print("Single scenario mode, multiple indicators possible")
        model = df.model[0]
        scenario = df.scenario[0]

        if len(mapdata.xrdataset.dims) > 2:
            map_array = mapdata.xrdataset.isel({gwl_name: 0}).reset_coords(drop=True)
            map_array = map_array.drop_vars(map_array.data_vars)
        else:
            map_array = xr.full_like(mapdata.xrdataset, np.nan).drop_vars(mapdata.xrdataset.data_vars)

        # needs to be outsite loop
        delayed_tasks = []

        df1 = df

        # use_dask not working here
        if use_dask:
            # Iterate through spatial indicators in DataSet by providing DataArrays
            for var_name in mapdata.xrdataset.data_vars:
                print(var_name)
                # Create delayed task for map_transform_gwl_wrapper
                delayed_map_transform = delayed(map_transform_gwl)(
                    df1, 
                    mapdata.xrdataset[var_name], 
                    years, 
                    var_name, 
                    map_array,
                    caution_checks=caution_checks,
                    include_orig_gwl=include_orig_gwl,
                    gwl_name=gwl_name,
                    drawdown_max=drawdown_max,
                    temp_min=temp_min,
                    temp_max=temp_max,
                    interpolation=interpolation,
                )
                delayed_tasks.append(delayed_map_transform)
        else:
            map_array = map_transform_gwl(
                df1, 
                mapdata.xrdataset, 
                years, 
                map_array, 
                caution_checks=caution_checks,
                include_orig_gwl=include_orig_gwl,
                gwl_name=gwl_name,
                drawdown_max=drawdown_max,
                temp_min=temp_min,
                temp_max=temp_max,
                interpolation=interpolation,
            )

        # use_dask not working here
        if use_dask:
            # Compute delayed tasks concurrently
            computed_results = dask.compute(*delayed_tasks)

            # Merge the computed results into map_array
            for result in computed_results:
                map_array.update(result)

        # map_array = map_array.assign_coords(
        #     coords={"lon": map_array.lon, "lat": map_array.lat, "year": years}
        # )
        map_array.attrs["model"] = model
        map_array.attrs["scenario"] = scenario
    return map_array


# =============================================================================
# CO2 functions
# =============================================================================
"""
These functions below can be used for basic processes relating CO2 and global
warming levels.
"""


def calculate_cumulative(ts, first_year, year, variable):
    """
    Input timeseries (e.g. CO2) and calculate the cumulative values through time.

    Parameters
    ----------
    ts : pandas.Dataframe()
        A pandas Dataframe in the IAMC wide format, with multi-index [model,scenario,region,variable,unit]
        and columns of [years]. Can be obtained by df.timeseries() from a pyam.IamDataFrame().
    first_year : int
        Year from which to start the cumulative calculation.
    year : int
        Last year until which to calculate cumulative.
    variable : str
        Name of the variable over which to calculate. e.g. Emissions|CO2

    Returns
    -------
    dfo : pandas.Dataframe()
        The input pandas.Dataframe() with additionally a new variable including the cumulative
        emissions through time along the timesteps.

    """
    dfo = (
        ts.apply(
            pyam.cumulative, raw=False, axis=1, first_year=first_year, last_year=year
        )
        .to_frame()
        .reset_index()
    )
    dfo["year"] = year
    dfo["variable"] = f"{variable}|Cumulative|from {first_year}"
    dfo.rename(columns={0: "value"}, inplace=True)
    return dfo


def prepare_cumulative(
    df,
    variable="Emissions|CO2",
    unit_in="Mt CO2/yr",
    unit_out="Gt CO2/yr",
    years=None,
    first_year=2020,
    last_year=2100,
    use_dask=False,
):
    """
    Prepares a pyam.IamDataFrame, by taking in variable (e.g. Emissions|CO2), filtering, optionally converting
    units (e.g. to Gt CO2), and then calculating the cumulative through time along the timesteps.

    Parameters
    ----------
    df : pyam.IamDataFrame
        DESCRIPTION.
    variable : str, optional
        DESCRIPTION. The default is 'Emissions|CO2'.
    unit_in : str, optional
        DESCRIPTION. The default is 'Mt CO2/yr'.
    years : list or np.array, optional
        If list or range is provided, cumulative emissions are calculated
        iteratively between the first value to each subsequent interval. If NONE,
        first_year and last)year are used.
    first_year : int, optional
        DESCRIPTION. The default is 2020.
    last_year : int, optional
        DESCRIPTION. The default is 2100.
    use_dask : boolean, optional
        If the datasets are very large, with many scenarios or high resolution
        of years, dask may perform faster. For smaller sets, keep as False.

    Returns
    -------
    df_cumlative :  pyam.IamDataFrame
        Returns pyam.IamDataFrame with cumulative variable (e.g. Emissions|CO2, Gt CO2/yr) through time,
        along each timestep.

    """

    if years is not None:
        years = list(years)
        if len(years) < 2:
            raise Exception("Error: years must be at least 2 items long")
        first_year = years[0]
        last_year = years[-1]
    else:
        years = [first_year, last_year]

    df_cumCO2 = pd.DataFrame()

    # filter and convert
    ts = (
        df.filter(
            variable=variable,
        )
        .convert_unit(unit_in, unit_out)
        .timeseries()
    )

    if use_dask:
        # Create a list of delayed computations
        dask_dfs = [
            delayed(calculate_cumulative)(ts, first_year, year, variable)
            for year in years
        ]

        # Compute the delayed computations and concatenate the results
        df_cumulative = dd.from_delayed(dask_dfs)
        df_cumlative = df_cumlative.compute()
    else:
        df_cumlative = pd.concat(
            [calculate_cumulative(ts, first_year, year, variable) for year in years]
        )

    df_cumlative["unit"] = unit_out

    return pyam.IamDataFrame(df_cumlative)


def co2togwl_simple(cum_CO2, regr=None):
    """
    Takes in vector of  CO2 values and calculates Global mean surface
    air temperature. Default is set up to use the linear regression between
    cumulative CO2 and the global mean surface air temperature (p50) from the
    IPCC AR6 Scenarios Database.

    Parameters
    ----------
    cum_CO2 : int, float, np.array, pandas.Series
        Value of cumulative CO2, from 2020 onwards, in Gt CO2.
    regr : dict, optional
        'slope' and 'intercept' values for linear regession. The default is None, in which
        case parameters from IPCC AR6 assessment are used. Provide {'slope': m,
        'intercept': x} to define own linear relationship.

    Returns
    -------
        Global mean surface air temperature.

    """

    # if isinstance(cum_CO2, pyam.IamDataFrame):

    cum_CO2 = np.array(cum_CO2)

    if regr == None:
        # use default AR6 paramters
        slope = 0.0005099869587542405
        intercept = 1.3024249460191835
    elif isinstance(regr, dict):
        # User provided parameters
        slope = regr["slope"]
        intercept = regr["intercept"]
    elif isinstance(regr, str):
        if regr == "AR6":
            # use default AR6 paramters
            slope = 0.0005099869587542405
            intercept = 1.3024249460191835
        # elif regr == "AR5":
        #     # use default AR6 paramters
        #     slope = 0.0043534
        #     intercept = 1.36555
        else:
            print("Warning: specification not recognized")
            raise Exception("Error: specification not recognized")
    elif isinstance(regr, pd.DataFrame):
        # Use linear regression based on pandas DataFrame of cumulative CO2 and
        # temperatures.
        x, y = regr.columns[0], regr.columns[1]
        slope, intercept, r, p, se = linregress(regr[x], regr[y])
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"r2: {r}")

    gwl = slope * cum_CO2 + intercept

    return gwl


# =============================================================================
# Plotting dashboards
# =============================================================================


def plot_maps_dashboard(
    ds,
    filename=None,
    indicators=None,
    year=2050,
    cmap="magma_r",
    shared_axes=True,
    clim=None,
    coastline=True,
    crs=None,
    features=None,
    layout_title=None,
):
    """
    From an xarray.DataSet of climate impact indicators through time for one IAM scenario (mode 2 in map_transform_gwl_wrapper),
    plot the indicators as an interactive html dashboard in a specified year.

    Parameters
    ----------
    ds : xarray.DataSet
        xarray.DataSet with dimensions of lat, lon and year, and each climate indicator as a variable.
    filename : str, optional
        Output filename as str ending in '.html' to save the file. Default will save as
        'maps_dashboard_{model}_{scenario}.html' in the current directory.
    indicators : list, optional
        A list of strings for the indicators in ds. Can be used to select only a subset of ass the
        indicators in ds.
    year : int, optional
        Year from which to plot data. The default is 2050.
    cmap : str, optional
        Continuous colormap from matplotlib.pyplot. The default is 'magma_r'.
    shared_axes : boolean, optional
        Whether zoom control automatically controls all axes. The default is True.
    clim : # not implemented yet!
    coastline : boolean, optional
        Show the coastlines on the map using Cartopy features. The default is True.
    crs : str, cartopy.Proj or pyproj.CRS, optional
        Must be either a valid crs or an reference to a `data.attr` containing a valid crs:
        Projection must be defined as a EPSG code, proj4 string, WKT string, cartopy CRS, pyproj.Proj, or pyproj.CRS.
        Provides the coordinate reference system to Cartopy. Requires Cartopy installation if used, and doesn't always work well.
        The default is None.
    features : List of strings, optional
        Whether to add Cartopy features to the map. The default is None.
        Available features include 'borders', 'coastline', 'lakes', 'land', 'ocean', 'rivers' and 'states'.


    Returns
    -------
    None. Output is saved to .html file as specified in filename.

    """

    if isinstance(features, type(None)):
        features = [
            "coastline",
        ]

    if isinstance(indicators, type(None)):
        indicators = list(ds.data_vars)
    elif isinstance(indicators, list):
        if not all(x in ds.data_vars for x in indicators):
            raise Exception(f"Error: not all items in indicators were found in ds.")
    elif isinstance(indicators, list) == False:
        raise Exception(f"Error: indicators must be of type list.")

    # Subset the dataset. Check dims and length

    ds = check_ds_dims(ds)

    # Check year input.
    if "year" in ds.dims:
        if year in ds.year:
            ds = ds.sel(year=year).squeeze()
        else:
            print(f"Warning: {year} not in original data, interpolating now")
            ds = ds.interp({"year": year})

    elif len(ds.dims) != 2:
        raise Exception(
            f"Error: Year not a dimension and more than 2 dimensions in dataset"
        )

    # Import cartopy if a crs is provided
    if not isinstance(crs, type(None)):
        try:
            import cartopy
        except:
            print("Error importing Cartopy")

    plot_list = []

    # Run loop through indicators (variables) and plot
    for i in indicators:

        new_plot = ds[i].hvplot(
            x="lon",
            y="lat",
            cmap=cmap,
            shared_axes=shared_axes,
            title=i,
            coastline=coastline,
            crs=crs,
            features=features,
            # check / add clim or vmin/vmax here
        )
        plot_list = plot_list + [new_plot]

    plot = hv.Layout(plot_list).cols(3)

    # Mode 1 - one model-scenario, multiple indicators
    if ("model" in ds.attrs.keys()) and ("scenario" in ds.attrs.keys()):
        model = ds.attrs["model"]
        scenario = ds.attrs["scenario"]
        title_2 = f": {model}, {scenario}"

    # Mode 2 - one indicator, multiple models/scenarios as the "variables"
    else:
        title_2 = f" {i}" # not sure if this works.

        
    if isinstance(layout_title, type(str)) == False:
        layout_title = f"Climate impacts in {year}: {title_2}"

    plot.opts(title=layout_title)

    # Plot - check filename - update below for mode 1/2
    if isinstance(filename, type(None)):
        filename = f"maps_dashboard_{model}_{scenario}.html"

    elif isinstance(filename, str):
        if (filename[-5:]) != ".html":
            raise Exception(f"filename {filename} must end with '.html'")

    else:
        raise Exception(f"filename must be string and end with '.html'")

    hvplot.save(plot, filename)
