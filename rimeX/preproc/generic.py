from pathlib import Path
from rimeX.config import CONFIG
from rimeX.logs import logger


class GenericIndicator:
    """Proposed generic Indicator class to inherit from
    """

    def __init__(self, name, simulations, ncvar=None,
                 spatial_aggregation=None,
                 comment=None, transform=None,
                 units="", title="", projection_baseline=None,
                 **kwargs):

        if len(kwargs):
            logger.debug(f"Unused arguments: {kwargs}")

        vars(self).update(kwargs)

        self.name = name
        self.simulations = simulations
        self.ncvar = ncvar or name

        self.spatial_aggregation = spatial_aggregation or CONFIG["preprocessing.regional.weights"]

        self.comment = comment
        self.transform = transform  # this refers to the baseline period and is only accounted for later on in the emulator (misnamed...)
        self._units = units
        self.title = title
        self.projection_baseline = projection_baseline or CONFIG["preprocessing.projection_baseline"]

    @property
    def units(self):

        if self.transform == "baseline_change_percent":
            return "%"

        if self._units:
            return self._units

        self._units = ""
        try:
            file = self.get_path(**self.simulations[0])
            import xarray as xa
            with xa.open_dataset(file, decode_times=False) as ds:
                v = ds[self.check_ncvar(ds)]
                for attr in ["units", "unit"]:
                    if attr in v.attrs:
                        units = v.attrs[attr]
                        logger.debug(f"{self.name} units found: {units}")
                        self._units = units
                        return self._units

                logger.warning(f"Cannot find units for {self.name}")
                return ""

        except Exception as e:
            logger.warning(e)
            logger.warning(f"Cannot find units for {self.name}. Leave empty.")
            return ""

        return units

    def check_ncvar(self, ds):
        """ we found an instance of sfcWind instead of sfcwind...
        """
        if self.ncvar not in ds:
            i = [k.lower() for k in ds].index(self.ncvar.lower())
            return list(ds)[i]
        return self.ncvar

    @classmethod
    def from_config(cls, name, **kw):
        cfg = CONFIG.get(f"indicator.{name}", {})
        copy = cfg.pop("_copy", None)
        if copy:
            cfg = {**CONFIG.get(f"indicator.{copy}", {}), **cfg}
        if name not in CONFIG["indicator"] and name not in CONFIG["isimip.variables"]:
            raise ValueError(f"Unknown indicator {name}")
        return cls(name, **{**cfg, **kw})

    def get_path(self, region=None, regional=False, regional_weight="latWeight", **simulation_specifiers):
        """returns the local file path for this indicator
        """
        raise NotImplementedError()


        # # one file for each region including admin boundaries
        # if region:
        #     assert regional_weight is not None
        #     regionfolders = [regional_weight, region]
        #     region_tag = f"_{region}_{regional_weight}".lower()
        #     ext = ".csv"

        # # one file for all regions
        # elif regional:
        #     region_tag = f"_regional_{regional_weight}".lower()
        #     regionfolders = []
        #     ext = ".csv"

        # # lon-lat file
        # else:
        #     region_tag = f""
        #     regionfolders = []
        #     ext = ".nc"

        # # otherwise create a new path separate from the ISIMIP database
        # basename = f"{climate_forcing.lower()}{ensemble_tag}_{climate_scenario}_{self.name}{region_tag}_{self.frequency}{timeslice_tag}"+ext
        # return Path(self.folder, self.name, climate_scenario, climate_forcing.lower(), *regionfolders, basename)


    def open_variable(self, region=None, regional=False, regional_weight="latWeight", **simulation_specifiers):
        """ that could be used to hide ncvar and check_ncvar
        """
        filepath = self.get_path(region, regional, regional_weight, **simulation_specifiers)

        if filepath.suffix == ".csv":
            import pandas as pd
            return pd.read_csv(filepath).to_xarray()

        from rimeX.compat import open_dataset
        with open_dataset(filepath) as ds:
            return ds[self.check_ncvar(ds)]