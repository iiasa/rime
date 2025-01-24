from pathlib import Path
from rimeX.config import CONFIG
from rimeX.logs import logger


class GenericIndicator:
    """Proposed generic Indicator class to inherit from
    """

    def __init__(self, name, simulations,
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


    def open_variable(self, xarray_kwargs={}, **kwargs):
        """ that could be used to hide ncvar and check_ncvar
        """
        filepath = self.get_path(**kwargs)

        if filepath.suffix == ".csv":
            import pandas as pd
            return pd.read_csv(filepath).to_xarray()

        from rimeX.compat import open_dataset
        with open_dataset(filepath, **xarray_kwargs) as ds:
            return ds[self.check_ncvar(ds)]