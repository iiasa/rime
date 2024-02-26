import xarray as xr


class RegionArray:
    def __init__(self, data_input):
        """Initialize the RegionArray class with either a file path or an xarray.Dataset."""
        if isinstance(data_input, xr.Dataset):
            self.dataset = data_input
        elif isinstance(data_input, str):
            self.dataset = xr.open_dataset(data_input)
        else:
            raise ValueError("Input must be an xarray.Dataset or a file path string compatible with xarray.open_dataset().")
        
        self._validate_and_adjust_dataset()

    def _validate_and_adjust_dataset(self):
        """Validate and adjust the dataset dimensions and coordinates as per requirements."""
        # Ensure dimensions are lowercase
        self.dataset = self.dataset.rename({dim: dim.lower() for dim in self.dataset.dims})

        # Check dimensions
        # Convert all dimension names to lowercase to standardize
        self.dataset = self.dataset.rename({dim: dim.lower() for dim in self.dataset.dims})

        # Check dimensions, now using lowercase to ensure case-insensitive comparison
        required_dims = ['region', 'year', 'gmt', 'ssp']
        missing_dims = [dim for dim in required_dims if dim not in self.dataset.dims]
        if missing_dims:
            raise ValueError(f"Dataset is missing required dimensions: {missing_dims}")

        # Ensure the length of 'gmt' is greater than 1
        if len(self.dataset['gmt']) <= 1:
            raise ValueError("The 'gmt' dimension must have a length greater than 1.")

        # Validate coordinates
        # Skip validation for 'year', only check if region and ssp are strings, gmt should be float
        if not all(isinstance(region, str) for region in self.dataset['region'].values):
            raise ValueError("All 'region' coordinates should be strings.")
        if not all(isinstance(gmt, float) for gmt in self.dataset['gmt'].values):
            raise ValueError("All 'gmt' coordinates should be floats.")
        if not all(isinstance(ssp, str) for ssp in self.dataset['ssp'].values):
            raise ValueError("All 'ssp' coordinates should be strings.")
        if not all(isinstance(year, (int, float) for ssp in self.dataset['ssp'].values):
            raise ValueError("All 'ssp' coordinates should be ints or floats.")
            
        # Ensure all 'gmt' coordinate values are > 0.5
        if not all(gmt > 0.5 for gmt in self.dataset['gmt'].values):
            raise ValueError("All 'gmt' coordinate values must be greater than 0.5.")
            
            
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
        else:
            raise ValueError("Input must be an xarray.Dataset, xarray.DataArray, or a file path string.")
    
        self._validate_and_adjust_dataset()
        self.tidy_rasterdata()  # Clean out any unwanted dimensions and coordinates


    def _validate_and_adjust_dataset(self):
        """Validate and adjust the dataset dimensions and coordinates as per requirements."""
        # Convert all dimension names to lowercase to standardize
        self.dataset = self.dataset.rename({dim: dim.lower() for dim in self.dataset.dims})

        # Check dimensions, now using lowercase to ensure case-insensitive comparison
        required_dims = ['lat', 'lon', 'gmt', 'ssp']
        missing_dims = [dim for dim in required_dims if dim not in self.dataset.dims]
        if missing_dims:
            raise ValueError(f"Dataset is missing required dimensions: {missing_dims}")

        # Ensure the length of 'gmt' is greater than 1
        if len(self.dataset['gmt']) <= 1:
            raise ValueError("The 'gmt' dimension must have a length greater than 1.")

        # Validate coordinates for 'lat' and 'lon', ensuring they are floats
        if not all(isinstance(lat, float) for lat in self.dataset['lat'].values):
            raise ValueError("All 'lat' coordinates should be floats.")
        if not all(isinstance(lon, float) for lon in self.dataset['lon'].values):
            raise ValueError("All 'lon' coordinates should be floats.")
        if not all(isinstance(gmt, float) for gmt in self.dataset['gmt'].values):
            raise ValueError("All 'gmt' coordinates should be floats.")
        if not all(isinstance(ssp, str) for ssp in self.dataset['ssp'].values):
            raise ValueError("All 'ssp' coordinates should be strings.")            
        
        # Ensure all 'gmt' coordinate values are > 0.5
        if not all(gmt > 0.5 for gmt in self.dataset['gmt'].values):
            raise ValueError("All 'gmt' coordinate values must be greater than 0.5.")

    def tidy_rasterdata(self):
        """Clean out any unwanted dimensions and coordinates."""
        dvs = self.dataset.data_vars
        self.dataset = self.dataset.rename({"threshold": "gmt"}) if "threshold" in self.dataset.dims else self.dataset
        self.dataset = self.dataset.set_index({"lon": "lon", "lat": "lat", "gmt": "gmt"}).reset_coords()
        self.dataset = self.dataset.drop_vars([x for x in self.dataset.data_vars if x not in dvs])
      
        
    def __repr__(self):
        """String representation of the dataset for quick inspection."""
        return repr(self.dataset)
