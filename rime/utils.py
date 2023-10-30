# utils.py
# Small helper functions





def fix_duplicate_temps(df, years):
    """
    Function that modifies GMT temperatures minutely, in case there are duplicates in the series.

    Parameters
    ----------
    df : pandas.DataFrame,
        a file from which to take transform and latitude objects

    Returns
    -------
    """

    vs = df[years]
    ld = len(df[years])
    lu = len(set(df[years]))

    if ld != lu:
        # print('')
        seen = set()
        vsn = []
        for x in vs:
            if x not in vsn:
                vsn.append(x)
            else:
                vsn.append(x + np.random.uniform(-0.005, 0.005))
        df[years] = vsn
    return df
	
	
def remove_ssp_from_ds(ds):
    """
    Preprocess input netCDF datasets to remove ssp from the variable names.
    Passed to the `preprocess` argument of xr.open_mfdataset()

    Parameters
    ----------
    ds : xarray.Dataset
        DESCRIPTION.

    Returns
    -------
    xarray.Dataset
        DESCRIPTION.

    """
    var = list(ds.keys())[0]
    short = f'{var.rsplit("_ssp")[0]}'

    return ds.rename({list(ds.keys())[0]: short})


def ssp_helper(dft, ssp_meta_col="Ssp_family", default_ssp="SSP2"):
    """
    Function to fill out and assign SSP to a meta column called Ssp_family. If
    there is no meta column with SSP information, automatically filled with
    default_ssp.

    Parameters
    ----------
    dft : pyam.IamDataFram
        input
    ssp_meta_col : Str, optional
        DESCRIPTION. The default is "Ssp_family".

    Returns
    -------
    None.

    """
    dft = np.round(dft.as_pandas()[pyam.IAMC_IDX + ["year", "value", ssp_meta_col]], 3)
    sspdic = {1.0: "SSP1", 2.0: "SSP2", 3.0: "SSP3", 4.0: "SSP4", 5.0: "SSP5"}
    dft[ssp_meta_col].replace(
        sspdic, inplace=True
    )  # metadata must have Ssp_family column. If not SSP2 automatically chosen
    dft.loc[dft[ssp_meta_col].isnull(), ssp_meta_col] = default_ssp

    return pyam.IamDataFrame(dft)
	
	


def check_ds_dims(ds):
    """
    Function to check the dimensions present in dataset before passing to plot maps

    Parameters
    ----------
    ds : xarray.Dataset
        If 2 dimensions, must be either x/y or lon/lat (former is renames to lon/lat). Third dimension can be 'year'. Otherwise errors are raised.

    Returns
    -------
    ds : xarray.Dataset with renamed dimensions, if necessary
        

    """
    if len(ds.dims) >= 3:
        if 'year' not in ds.dims:
            raise ValueError("The dataset contains 3 or more dimensions, but 'year' dimension is missing.")

    if 'lat' in ds.dims and 'lon' in ds.dims:
        return ds
    elif 'x' in ds.dims and 'y' in ds.dims:
        ds = ds.rename({'x': 'lat', 'y': 'lon'})
        return ds
    else:
        raise ValueError("The dataset does not contain 'lat' and 'lon' or 'x' and 'y' dimensions.")
