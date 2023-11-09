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
import numpy as np
import pandas as pd
import pyam
from scipy.stats import linregress
import xarray as xr

from rime.utils import check_ds_dims


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
    dfx,
    dsi,
    prefix_indicator="RIME|",
    ssp_meta_col="Ssp_family",
):
    """
    Takes in a set of scenarios of GMT and a dataset of climate impacts by GWL,
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
    # gmt_below : float, optional
    #     DESCRIPTION. The default is 1.5. Assign all gmt values below gmt_below to the value of gmt_below (in case not enough data)

    Returns
    -------
    idf : pyam.IamDataFrame
        pyam.IamDataFrame with variables for the impact indicators for each scenario(s).

    """

    model = dfx["model"]
    scenario = dfx["scenario"]

    ssp = dfx[ssp_meta_col] #<<< isue here?

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





# =============================================================================
# Maps
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
        # WORKS in multi scenario mode, but not single scenario!!!!!
        # if len(mapdata.data_vars) >1:
        #     try:
        #         mapdata = mapdata.rename_vars({gmt_name: "gmt"})
        #     except ValueError:
        #         print("")

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
    gmt_name="gmt",
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
    



    if len(df.index) > 1:
        if len(mapdata.dims)>2:
            map_array = mapdata.isel({gmt_name:0}).reset_coords(drop=True)
            map_array = map_array.drop_vars(map_array.data_vars)
        else:
            map_array = xr.full_like(mapdata, np.nan).drop_vars(mapdata.data_vars)
        
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
        model = df.model[0]
        scenario = df.scenario[0]
        
        
        
        delayed_tasks = []
        if len(mapdata.dims)>2:
            map_array = mapdata.isel({gmt_name:0}).reset_coords(drop=True)
            map_array = map_array.drop_vars(map_array.data_vars)
        else:
            map_array = xr.full_like(mapdata, np.nan).drop_vars(mapdata.data_vars)
        

        # Iterate through spatial indicators in DataSet
        for var_name in mapdata.data_vars:
            print(var_name)
            df1 = df
            if use_dask:
                # Create delayed task for map_transform_gmt_dask
                delayed_map_transform = delayed(map_transform_gmt)(
                    df1, mapdata[var_name], var_name, years, map_array
                )
                delayed_tasks.append(delayed_map_transform)
            else:
                map_array = map_transform_gmt(df1, mapdata[var_name], var_name, years, map_array)
        if use_dask:
            # Compute delayed tasks concurrently
            computed_results = dask.compute(*delayed_tasks)

            # Merge the computed results into map_array
            for result in computed_results:
                map_array.update(result)

    map_array = map_array.assign_coords(
        coords={"lon": map_array.lon, "lat": map_array.lat, "year": years}
    )
    map_array.attrs['model'] = model
    map_array.attrs['scenario'] = scenario
    return map_array


def calculate_cumulative_CO2(ts, first_year, year, variable):
    """
    

    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    first_year : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    variable : TYPE
        DESCRIPTION.

    Returns
    -------
    dfo : TYPE
        DESCRIPTION.

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


def prepare_cumCO2(
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
    Prepares a dataframe, by taking in CO2 emissions, filtering, converting
    units to Gt CO2, and then calculating cumulative CO2.

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
    df :  pyam.IamDataFrame
        Returns df with cumulative CO2 (Gt CO2/yr) through time.

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
        .convert_unit(unit_in, "Gt CO2/yr")
        .timeseries()
    )

    if use_dask:
        # Create a list of delayed computations
        dask_dfs = [
            delayed(calculate_cumulative_CO2)(ts, first_year, year, variable)
            for year in years
        ]

        # Compute the delayed computations and concatenate the results
        df_cumCO2 = dd.from_delayed(dask_dfs)
        df_cumCO2 = df_cumCO2.compute()
    else:
        df_cumCO2 = pd.concat(
            [calculate_cumulative_CO2(ts, first_year, year, variable) for year in years]
        )

    df_cumCO2["unit"] = unit_out

    return pyam.IamDataFrame(df_cumCO2)


def co2togmt_simple(cum_CO2, regr=None):
    """
    Takes in vector of  CO2 values and calculates Global mean surface
    air temperature (p50).
    Parameters
    ----------
    cum_CO2 : int, float, np.array, pandas.Series
        Value of cumulative CO2, from 2020 onwards, in Gt CO2.
    regr : dict, optional
        'slope' and 'intercept' values for line. The default is None, in which
        case parameters from AR6 assessment are used. Provide {'slope': m,
        'intercept': x} to define own linear relationship.

    Returns
    -------
    Global mean surface air temperature (p50).

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
        # Use linear regression based on DataFrame
        x, y = regr.columns[0], regr.columns[1]
        slope, intercept, r, p, se = linregress(regr[x], regr[y])
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"r2: {r}")

    gmt = slope * cum_CO2 + intercept

    return gmt

#%%
def plot_maps_dashboard(ds, filename=None, indicators=None, year=2050, 
                        cmap='magma_r', shared_axes=True, clim=None, 
                        coastline=True, crs=None, features=None, layout_title=None):
    """
    

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    filename : TYPE, optional
        DESCRIPTION. The default is None.
    indicators : TYPE, optional
        DESCRIPTION. The default is None.
    year : int, optional
        Year from which to plot data. The default is 2050.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'magma_r'.
    shared_axes : TYPE, optional
        DESCRIPTION. The default is True.
    clim : TYPE, optional
        DESCRIPTION. The default is None.
    coastline : TYPE, optional
        DESCRIPTION. The default is True.
    crs : must be either a valid crs or an reference to a `data.attr` containing a valid crs: Projection must be defined as a EPSG code, proj4 string, WKT string, cartopy CRS, pyproj.Proj, or pyproj.CRS., optional
        DESCRIPTION. Provides the coordinate reference system to Cartopy. 
        Requires Cartopy installation if used, and doesn't always work well. 
        The default is None.
    features : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if isinstance(features, type(None)):
        features = ['coastline', ]
    
    if isinstance(indicators, type(None)):
        indicators = list(ds.data_vars)
    elif isinstance(indicators, list):
        if not all(x in ds.data_vars for x in indicators):
            raise Exception(f"Error: not all items in indicators were found in ds.")
    elif isinstance(indicators, list)==False:
        raise Exception(f"Error: indicators must be of type list.")

    
    # Subset the dataset. Check dims and length
    
    ds = check_ds_dims(ds)
    
    
    # Check year input.
    if 'year' in ds.dims:
        if year in ds.year:
            ds = ds.sel(year=year).squeeze()
        else:
            print("Warning: year not in original data, interpolating now")
            ds = ds.interp({'year':year})
        
    elif len(ds.dims) != 2:
        raise Exception(f"Error: Year not a dimension and more than 2 dimensions in dataset")
        

    plot_list = []

    # Import cartopy if a crs is provided
    if not isinstance(crs, type(None)):
        try:
            import cartopy
        except:
            print('Error importing Cartopy')
            

    # Run loop through indicators (variables) and plot
    for i in indicators:
        
        new_plot = ds[i].hvplot(x='lon', y='lat', cmap=cmap, 
                                shared_axes=shared_axes, title=i, 
                                coastline=coastline, crs=crs, 
                                features=features)
        plot_list = plot_list + [new_plot]

    plot = hv.Layout(plot_list).cols(3)
    

    
    if ('model' in ds.attrs.keys()) and ('scenario' in ds.attrs.keys()):
        model = ds.attrs['model']
        scenario = ds.attrs['scenario']
        # layout title

    else:
        model = 'unknown'
        scenario = 'scenario'
    if isinstance(layout_title, type(None))==False:
        layout_title = f'Climate impacts in {year} ({model}, {scenario})'

    
    plot.opts(title=layout_title)
    
    # Plot - check filename
    if isinstance(filename, type(None)):
        filename = f'maps_dashboard_{model}_{scenario}.html'
    
    elif isinstance(filename, str):
        if (filename[-5:]) != '.html':
            raise Exception(f"filename {filename} must end with '.html'")
                
    else:
        raise Exception(f"filename must be string and end with '.html'")
        
    hvplot.save(plot, filename)


#%%

# # features = ['ocean']
# indicators = ['cdd','dri']

# filename = 'test_map.html'
# plot_maps_dashboard(map_out_MI, indicators =indicators,
#                     filename=filename,  year=2055, cmap='magma_r', 
#                     shared_axes=True, clim=None, )
# os.startfile(filename)



#%%