from pathlib import Path
from rimeX.config import CONFIG
from rimeX.logs import logger
import xarray as xa


class GenericIndicator:
    """Proposed generic Indicator class to inherit from
    """

    def __init__(self, name, simulations, frequency="monthly",
                 transform=None, units="", title="", projection_baseline=None, **kwargs):

        if len(kwargs):
            logger.debug(f"Unused arguments: {kwargs}")

        vars(self).update(kwargs)

        self.name = name
        self._simulations = simulations
        self.transform = transform  # this refers to the baseline period and is only accounted for later on in the emulator (misnamed...)
        self.projection_baseline = projection_baseline
        self._units = units
        self.title = title
        self.frequency = frequency

    @property
    def units(self):
        if "percent" in self.transform:
            return "%"
        return self._units

    @property
    def simulations(self):
        return self._simulations

    @classmethod
    def from_config(cls, name, **kw):
        raise NotImplementedError()

    def get_path(self, region: str = None, regional: bool = False, regional_weight: str = "latWeight", **simulation_specifiers) -> Path:
        """Returns the local file path for this indicator

        The default is to return the gridded lon / lat file.
        Regional-average files are obtained by specifying `region` or `regional`, as well as `regional_weight` (See below)

        Parameters
        ----------
        region : str, optional
            If provided, the file path is for the region. The designated file contains all sub-regions for that region.

        regional : bool, optional
            If True, return a file that contain all regions (but no sub-regions) -- we normally don't use this for the CIE

        regional_weight : str, optional
            The weight for the region. The default is "latWeight". Other possible weights include "gdp2005" and "pop2005".

        **simulation_specifiers
            The destructured dict that could be any item from the `simulations` list.
            For ISIMIP simulations, it contains `climate_forcing`, `climate_scenario` and sometimes `model` (for the impact model)
            For CMIP simulations that would be `model` (? CHECK), `experiment`, etc...

        Returns
        -------
        a Path object
        """
        raise NotImplementedError()


    ## The methods below could be separate functions. They are intended to be as general as possible
    ## To define a custom load function, it is possible to override the open_simulation method method
    ## Or the lower-level functions _load_csv_file and _load_nc_file

    @staticmethod
    def _load_csv_file(filepath, metadata={}) -> xa.DataArray:
        import pandas as pd
        df = pd.read_csv(filepath, index_col=0)
        metadata = metadata.copy()
        name = metadata.pop("name", None)

        return xa.DataArray(df,
                # make sure we have dates as index (and not just years, cause the calling function needs dates)
                coords=[pd.to_datetime(df.index.astype(str)), df.columns],
                dims=["time", "region"], name=name).assign_attrs(metadata)

    def _check_ncvar(self, ds):
        """
        it can be tricky to find the proper netCDF file name : ncvar and indicator names generally differ,
        and at some point during the processing it may have been renamed from ncvar to indicator name.
        Even ncvar has been found to vary across source ISIMIP files (e.g. sfcwind and sfcWind)
        """
        ncvars = list(ds)

        # simple when only one non-coordinate variable is present:
        if len(ncvars) == 1:
            return ncvars[0]

        # otherwise use the indicator name or the ncvar attribute, if any
        candidate_vars = [self.name]
        if hasattr(self, "ncvar"):
            candidate_vars.append(self.ncvar)

        # compare case-insensitive names (e.g. sfcwind and sfcWind exist)
        candidate_vars = map(str.lower, candidate_vars)
        ncvars_lower = map(str.lower, ncvars)

        intersection = set(ncvars_lower).intersection(candidate_vars)
        if not intersection:
            raise ValueError(f"Could not find a variable for {self.name} in {ncvars}")
        elif len(intersection) > 1:
            raise ValueError(f"Multiple variables found for {self.name} in {ncvars}: {intersection}")
        assert len(intersection) == 1

        v = intersection.pop()
        return ncvars[ncvars_lower.index(v)]

    def _load_nc_file(self, filepath, metadata={}, ncvar=None, xarray_kwargs={}) -> xa.DataArray:
        """ Basic function that is superceded in download_isimip.Indicator because it does not account for variation in ncvar (e.g. sfcwind and sfcWind)
        """
        from rimeX.compat import open_dataset
        with open_dataset(filepath, **xarray_kwargs) as ds:
            if ncvar is None:
                ncvar = self._check_ncvar(ds)
            return ds[ncvar].rename(self.name).assign_attrs(metadata)

    def open_simulation(self, xarray_kwargs={}, **simu) -> xa.DataArray:
        """ That's the main function
        """
        filepath = self.get_path(**simu)
        metadata = dict(units=self.units, name=self.name, **simu)

        if filepath.suffix == ".csv":
            return self._load_csv_file(filepath, metadata=metadata)

        if filepath.suffix == ".nc":
            return self._load_nc_file(filepath, xarray_kwargs=xarray_kwargs, metadata=metadata)

        raise NotImplementedError(filepath.suffix)