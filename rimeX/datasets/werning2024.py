import os
import tempfile
from pathlib import Path
import subprocess as sp

from rimeX.config import logger
from rimeX.datasets.manager import get_datapath, register_dataset

NAME = __name__.split(".")[-1]
DATA = get_datapath(NAME)
DOI = "10.5281/zenodo.7971429"


require_table_avoided_impacts = register_dataset(f"{NAME}/table_output_avoided_impacts", 
    url="https://zenodo.org/records/10868066/files/table_output_avoided_impacts.zip?download=1", doi=DOI)

require_table_output_climate_exposure = register_dataset(f"{NAME}/table_output_climate_exposure", 
    url="https://zenodo.org/records/10868066/files/table_output_climate_exposure.zip?download=1", doi=DOI)

require_precipitation = register_dataset(f"{NAME}/precipitation", 
    url="https://zenodo.org/records/10868066/files/precipitation.zip?download=1", doi=DOI)

require_temperature = register_dataset(f"{NAME}/temperature", 
    url="https://zenodo.org/records/10868066/files/temperature.zip?download=1", doi=DOI)

require_air_pollution = register_dataset(f"{NAME}/air_pollution", 
    url="https://zenodo.org/records/10868066/files/air_pollution.zip?download=1", doi=DOI)

require_energy = register_dataset(f"{NAME}/energy", 
    url="https://zenodo.org/records/10868066/files/energy.zip?download=1", doi=DOI)

require_hydrology = register_dataset(f"{NAME}/hydrology", 
    url="https://zenodo.org/records/10868066/files/hydrology.zip?download=1", doi=DOI)

require_land = register_dataset(f"{NAME}/land", 
    url="https://zenodo.org/records/10868066/files/land.zip?download=1", doi=DOI)