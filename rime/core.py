import numpy as np
import pandas as pd
import pyam
import xarray as xr
from rime.utils import ssp_helper  # Import ssp_helper from rime.utils


class RegionArray:
    def __init__(self, data_input):
        """Initialize the RegionArray class with either a file path, list of file paths or an xarray.Dataset."""
        if isinstance(data_input, xr.Dataset):
            self.dataset = data_input
        elif isinstance(data_input, str):
            self.dataset = xr.open_dataset(data_input)
        elif isinstance(data_input, list):
            # list of filepaths
            self.dataset = xr.open_mfdataset(data_input)            
        else:
            raise ValueError("Input must be an xarray.Dataset, a file path string or a list of file path strings compatible with xarray.open_dataset() or xarray.open_mfdataset().")
        
        self._validate_and_adjust_dataset()

    def _validate_and_adjust_dataset(self):
        """Validate and adjust the dataset dimensions and coordinates as per requirements."""
        # Check dimensions
        # Convert all dimension names to lowercase to standardize
        self.dataset = self.dataset.rename({dim: dim.lower() for dim in self.dataset.dims})

        self.dataset = self.dataset.rename({"gmt": "gwl"}) if "gmt" in self.dataset.dims else self.dataset
        
        # Check dimensions, now using lowercase to ensure case-insensitive comparison
        required_dims = ['region', 'year', 'gwl', 'ssp']
        missing_dims = [dim for dim in required_dims if dim not in self.dataset.dims]
        if missing_dims:
            raise ValueError(f"Dataset is missing required dimensions: {missing_dims}")

        # Ensure the length of 'gwl' is greater than 1
        if len(self.dataset['gwl']) <= 1:
            raise ValueError("The 'gwl' dimension must have a length greater than 1.")

        # Validate coordinates
        # Skip validation for 'year', only check if region and ssp are strings, gwl should be float
        if not all(isinstance(region, str) for region in self.dataset['region'].values):
            raise ValueError("All 'region' coordinates should be strings.")
        if not all(isinstance(gwl, float) for gwl in self.dataset['gwl'].values):
            raise ValueError("All 'gwl' coordinates should be floats.")
        if not all(isinstance(ssp, str) for ssp in self.dataset['ssp'].values):
            raise ValueError("All 'ssp' coordinates should be strings.")
        if not all(isinstance(year, (int, np.int32, np.int64, float, np.float32, np.float64)) for year in self.dataset['year'].values):
            raise ValueError("All 'year' coordinates should be ints or floats.")
            
        # Ensure all 'gwl' coordinate values are > 0.5
        if not all(gwl > 0.5 for gwl in self.dataset['gwl'].values):
            raise ValueError("All 'gwl' coordinate values must be greater than 0.5.")
            
            
    def __repr__(self):
        """String representation of the dataset for quick inspection."""
        return repr(self.dataset)



class RasterArray:
    def __init__(self, data_input):
        """Initialize the RasterArray class with a file path, xarray.Dataset, or xarray.DataArray."""
        if isinstance(data_input, xr.Dataset):
            self.dataset = data_input
        elif isinstance(data_input, xr.DataArray):
            # Convert DataArray to Dataset
            self.dataset = data_input.to_dataset()
        elif isinstance(data_input, str):
            self.dataset = xr.open_dataset(data_input)
        elif isinstance(data_input, list):
            from rime.utils import remove_ssp_from_ds

            self.dataset = xr.open_mfdataset(data_input, preprocess=remove_ssp_from_ds, combine="nested", concat_dim="gwl"
)
        else:
            raise ValueError("Input must be an xarray.Dataset, xarray.DataArray, a file path string, or a list of file path strings.")
    
        self.tidy_rasterdata()  # Clean out any unwanted dimensions and coordinates
        self._validate_and_adjust_dataset()
        self.xrdataset = self.dataset


    def _validate_and_adjust_dataset(self):
        """Validate and adjust the dataset dimensions and coordinates as per requirements."""
        # Convert all dimension names to lowercase to standardize
        self.dataset = self.dataset.rename({dim: dim.lower() for dim in self.dataset.dims})

        # Check dimensions, now using lowercase to ensure case-insensitive comparison
        required_dims = ['lat', 'lon', 'gwl',]
        missing_dims = [dim for dim in required_dims if dim not in self.dataset.dims]
        if missing_dims:
            raise ValueError(f"Dataset is missing required dimensions: {missing_dims}")

        # Ensure the length of 'gwl' is greater than 1
        if len(self.dataset['gwl']) <= 1:
            raise ValueError("The 'gwl' dimension must have a length greater than 1.")

        # Validate coordinates for 'lat' and 'lon', ensuring they are floats
        if not all(isinstance(lat, (float, np.float32, np.float64)) for lat in self.dataset['lat'].values):
            raise ValueError("All 'lat' coordinates must be floats.")
        if not all(isinstance(lon, (float, np.float32, np.float64)) for lon in self.dataset['lon'].values):
            raise ValueError("All 'lon' coordinates must be floats.")
        if not all(isinstance(gwl, (float, np.float32, np.float64)) for gwl in self.dataset['gwl'].values):
            raise ValueError("All 'gwl' coordinates must be floats.")
        # if not all(isinstance(ssp, str) for ssp in self.dataset['ssp'].values):
            # raise ValueError("All 'ssp' coordinates should be strings.")            
        
        # Ensure all 'gwl' coordinate values are > 0.5
        if not all(gwl > 0.5 for gwl in self.dataset['gwl'].values):
            raise ValueError("All 'gwl' coordinate values must be greater than 0.5.")

    def tidy_rasterdata(self):
        """Clean out any unwanted dimensions and coordinates."""
        dvs = self.dataset.data_vars
        self.dataset = self.dataset.rename({"threshold": "gwl"}) if "threshold" in self.dataset.dims else self.dataset
        self.dataset = self.dataset.rename({"threshold": "gwl"}) if "threshold" in self.dataset.coords else self.dataset
        self.dataset = self.dataset.rename({"gmt": "gwl"}) if "gmt" in self.dataset.dims else self.dataset
        print(self.dataset.dims)
        self.dataset = self.dataset.set_index({"lon": "lon", "lat": "lat", "gwl": "gwl"}).reset_coords()
        self.dataset = self.dataset.sortby("gwl")
        self.dataset = self.dataset.drop_vars([x for x in self.dataset.data_vars if x not in dvs])
      
        
    def __repr__(self):
        """String representation of the dataset for quick inspection."""
        return repr(self.dataset)

class GMTPathway:
    """
    A class for processing global mean temperature pathways from input data files or existing dataframes, 
    with a focus on filtering and analyzing temperature-related variables.

    The class supports input data in the form of a CSV or Excel file path, or an existing `pyam.IamDataFrame`. 
    It determines the appropriate temperature variables to filter on, either automatically (if there's only one unique variable in the input data) 
    or based on user specification. The class also uses the `ssp_helper` function from `rime.utils` for further data processing.

    Attributes:
        data_input (str or pyam.IamDataFrame): The input data source, which can be a file path (CSV or Excel) or an existing `pyam.IamDataFrame`.
        temperature_variable (str or list of str, optional): The name(s) of the temperature variable(s) to filter on. 
            If not provided and the input data contains multiple variables, an error is raised.
        ssp_meta_col (str, optional): The metadata column name used in the `ssp_helper` function to identify the SSP of scenarios. Defaults to "Ssp_family".
        default_ssp (str, optional): The default SSP scenario to use in the `ssp_helper` function if none is specified in the data. Defaults to "SSP2". This is needed when using exposure and vulnerability data.
        df (pyam.IamDataFrame): The processed `IamDataFrame` after filtering based on temperature variables and applying the `ssp_helper` function.

    Methods:
        _load_dataframe(): Loads the input data into a `pyam.IamDataFrame`, automatically detecting the file type if a file path is provided.
        _ensure_temperature_variable(): Validates the existence of specified temperature variables in the dataframe, raising an error if any are missing.
        _process_dataframe(): Filters the dataframe based on specified temperature variables and applies the `ssp_helper` function for further processing.

    Raises:
        ValueError: If the input data type is unsupported, or if specified temperature variables are not found in the dataframe variables.
    
    Example:
        >>> gmt_pathway = GMTPathway('path/to/data.csv', temperature_variable=['Surface Temperature'], ssp_meta_col='Scenario Family', default_ssp='SSP3')
        >>> print(gwl_pathway.df)  # Display the processed IamDataFrame
    """    
    def __init__(self, data_input, temperature_variable=None, ssp_meta_col="Ssp_family", default_ssp="SSP2"):
        self.data_input = data_input
        self.temperature_variable = temperature_variable
        self.ssp_meta_col = ssp_meta_col
        self.default_ssp = default_ssp
        self.df = self._load_dataframe()
        self._process_dataframe()
        self._ensure_temperature_variable()
        self.index = self.df.index
        self.meta = self.df.meta

        # if len(self.df.meta==0):
            # raise ValueError("Empty dataframe.")

    def _load_dataframe(self):
        """Load the input file into a pyam.IamDataFrame."""
        if isinstance(self.data_input, pyam.IamDataFrame):
            return self.data_input
        elif self.data_input.endswith(('.xlsx', '.xls', 'csv')):
            return pyam.IamDataFrame(self.data_input)
        else:
            raise ValueError("Unsupported type. Please provide a pyam.IamDataFrame, CSV or Excel file.")
   
    def _process_dataframe(self):
        """Filter the dataframe based on the temperature variable and apply the ssp_helper function."""
        print(self.temperature_variable)

        unique_vars = self.df.variable
        if isinstance(self.temperature_variable, (str, list)):
            print(f'Temperature_variable(s) provided: {self.temperature_variable}')

        # 1 variable provided
        # Determine temperature_variable based on the unique values in the 'variable' column
        elif len(unique_vars) == 1 and self.temperature_variable is None:
            self.temperature_variable = unique_vars
            print('Only one variable detected - assuming that this is a temperature pathway')
       
        # more than 1 provided and none specified - raise error
        elif len(unique_vars) > 1 and self.temperature_variable is None:
            print(unique_vars)
            print('case21')
            raise ValueError("Multiple variables found. Please specify the temperature_variable(s) as str or list of strings.")
        else:
            raise ValueError("iiidunnoo1")

        if self.temperature_variable is not None:

            filtered_df = self.df.filter(variable=self.temperature_variable)
            self.df = ssp_helper(filtered_df, self.ssp_meta_col, self.default_ssp)  # Use imported ssp_helper
            # print('case31')

        else:
            print(len(self.df.variable))
            # print('case41')
            raise ValueError("fucked1")

    def _ensure_temperature_variable(self):
        variables = self.df.variable
        if isinstance(self.temperature_variable, list):
            missing_vars = [var for var in self.temperature_variable if var not in variables]
            if missing_vars:
                raise ValueError(f"Specified temperature variables {missing_vars} not found in dataframe variables.")
        elif isinstance(self.temperature_variable, str):
            if self.temperature_variable not in variables:
                raise ValueError(f"Specified temperature variable '{self.temperature_variable}' not found in dataframe variables.")
        else:
            raise ValueError("temperature_variable must be a string or a list of strings.")

    def __repr__(self):
        """String representation of the processed IamDataFrame for quick inspection."""
        return repr(self.df)


