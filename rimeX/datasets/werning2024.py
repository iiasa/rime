import os
import tempfile
from pathlib import Path
import subprocess as sp

from rimeX.config import logger
from rimeX.datasets.manager import get_datapath, require_dataset

NAME = __name__.split(".")[-1]
DATA = get_datapath(NAME)
DOI = "10.5281/zenodo.7971429"

def require_table_avoided_impacts():
    return require_dataset(f"{NAME}/table_output_avoided_impacts", 
        url="https://zenodo.org/records/10868066/files/table_output_avoided_impacts.zip?download=1", doi=DOI)

def require_table_output_climate_exposure():
    return require_dataset(f"{NAME}/table_output_climate_exposure", 
        url="https://zenodo.org/records/10868066/files/table_output_climate_exposure.zip?download=1", doi=DOI)

def require_precipitation():
    return require_dataset(f"{NAME}/precipitation", 
        url="https://zenodo.org/records/10868066/files/precipitation.zip?download=1", doi=DOI)

def require_temperature():
    return require_dataset(f"{NAME}/temperature", 
        url="https://zenodo.org/records/10868066/files/temperature.zip?download=1", doi=DOI)

def require_air_pollution():
    return require_dataset(f"{NAME}/air_pollution", 
        url="https://zenodo.org/records/10868066/files/air_pollution.zip?download=1", doi=DOI)

def require_energy():
    return require_dataset(f"{NAME}/energy", 
        url="https://zenodo.org/records/10868066/files/energy.zip?download=1", doi=DOI)

def require_hydrology():
    return require_dataset(f"{NAME}/hydrology", 
        url="https://zenodo.org/records/10868066/files/hydrology.zip?download=1", doi=DOI)

def require_land():
    return require_dataset(f"{NAME}/land", 
        url="https://zenodo.org/records/10868066/files/land.zip?download=1", doi=DOI)

def download():
    "bulk download"
    require_table_avoided_impacts()
    require_table_output_climate_exposure()
    require_precipitation()
    require_temperature()
    require_air_pollution()
    require_energy()
    require_hydrology()
    require_land()