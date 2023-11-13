# RIME - Rapid Impact Model Emulator

2023 IIASA

[![latest](https://img.shields.io/github/last-commit/iiasa/CWatM)](https://github.com/iiasa/CWatM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
[![license](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://choosealicense.com/licenses/gpl-3.0/)

![RIME_logo](https://github.com/iiasa/rime/blob/main/assets/RIME_logo2.png?raw=true)  

## Overview  
------------------  

**RIME** is a lightweight software tool for using global warming level approaches to link climate impacts to Integrated Assessment Models (IAMs).

When accompanied by climate impacts data (table and/or maps), RIME can be used to take a global mean temperature timeseries (e.g. from an IAM or climate model like FaIR/MAGICC), and return tables and maps of climate impacts through time consistent with the warming of the scenario.  

There are two key use-cases for the RIME approach:  
1. **Post-process**: Estimating a suite of climate impacts from a global emissions or temperature scenario.  
2. **Input**: Reformulating climate impacts data to be used as an input to an integrated assessment model scenario.  

![RIME_use_cases](https://github.com/iiasa/rime/blob/main/assets/rime_use_cases.jpg?raw=true)  


## Core files

### `rime_functions.py` 
Contains the key functions that can be used to process data. 

### `utils.py`
A collection of helper functions related to data processing, used within functions and as standalone, if needed.

### `process-config.py` 
A script to host a large number of configurable settings for running the software on datasets.
Needs to be imported at the beginning of a script, e.g. `from process_config import *`.  
Settings for `Dask` could be configured in here. 

## Example processing and workflow scripts

### `generate_aggregated_inputs.py` 
Pre-processing of tabular impacts data of exposure by GWL, into netcdf datasets that will be used in emulation. Only needs to run once to pre-process the impacts data. Only required if working with IAMC table impacts data.

### `process_tabledata.py` 
Example script that takes input table of emissions scenarios with global temperature timeseries, and output tables of climate impacts data in IAMC format. Can be done for multiple scenarios and indicators at a time. 

### `process_maps.py`  
Example script that takes input table of emissions scenarios with global temperature timeseries, and output maps of climate impacts through time as netCDF. Ouptut netCDF can be specified for either for 1 scenario and multiple climate impacts, or multiple scenarios for 1 indicator.

### `pp_combined example.ipynb`
Example jupyter notebook that demonstrates methods of processing both table and map impacts data for IAM scenarios.


## Installation

At command line, navigate to the directory where you want the installation, e.g. your Github folder.  

	git clone https://github.com/iiasa/rime.git

Change to the rime folder and install the package including the requirements.  

	pip install --editable .

## Further information
This package is in a pre-release mode, currently work in progress, under-going testing and not formally published.  

Examples provided use climate impacts data that is also in a pre-release stage [(Werning et al. 2023)](https://zenodo.org/records/8134869), currently hosted on the [Climate Solutions Explorer](https://www.climate-solutions-explorer.eu/).

