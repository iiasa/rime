# RIME - Rapid Impact Model Emulator

2023 IIASA

[![latest](https://img.shields.io/github/last-commit/iiasa/RIME)](https://github.com/iiasa/RIME)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
[![license](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://choosealicense.com/licenses/gpl-3.0/)

![RIME_logo](https://github.com/iiasa/rime/blob/main/assets/RIME_logo2.png?raw=true)  

## Overview  
------------------  

**RIME** is a lightweight software tool for using global warming level approaches to link climate impacts data to Integrated Assessment Models (IAMs).

When accompanied by climate impacts data (table and/or maps), RIME can be used to take a global mean temperature timeseries (e.g. from an IAM or climate model like [FaIR](https://github.com/OMS-NetZero/FAIR)/[MAGICC](https://live.magicc.org/)), and return tables and maps of climate impacts through time consistent with the warming of the scenario.  

There are two key use-cases for the RIME approach:  
1. **Post-process**: Estimating a suite of climate impacts from a global emissions or temperature scenario.  
2. **Input**: Reformulating climate impacts data to be used as an input to an integrated assessment model scenario.  

![RIME_use_cases](https://github.com/iiasa/rime/blob/main/assets/rime_use_cases.jpg?raw=true)  

**RIME** is *Rapid*! *It's in the name...*
 - RIME is intended and designed to be rapid. It uses [Xarray](https://github.com/pydata/xarray) and [Dask](https://github.com/dask/dask) for parallelized processing and lazy reading of big data. Processing climate impacts data takes **in the order of seconds per scenario** on a desktop computer.
 - RIME is intended and designed with the [IAMC](https://www.iamconsortium.org/) and [ISIMIP](https://www.isimip.org) communities in mind. It uses [Pyam](https://github.com/iamconsortium/pyam) for processing IAM scenarios and follows community data formats.

![image](https://github.com/iiasa/rime/assets/17701232/7f3fec80-ab5a-468b-99d8-e759628f7542)


## Core files

### [`rime_functions.py`](https://github.com/iiasa/rime/blob/main/rime/rime_functions.py)   
Contains the key functions that can be used to process data and generate outputs.

### [`utils.py`](https://github.com/iiasa/rime/blob/main/rime/utils.py)  
A collection of helper functions related to data processing, used within functions and as standalone, if needed.

### [`process_config.py`](https://github.com/iiasa/rime/blob/main/rime/process_config.py)  
A script to host a large number of configurable settings for running the software on datasets.
Needs to be imported at the beginning of a script, e.g. `from process_config import *`.  
Settings for `Dask`, filepaths and data directories should be configured in here for your local configuration. 

## Example processing and workflow scripts

### [`generate_aggregated_inputs.py`](https://github.com/iiasa/rime/blob/main/rime/generate_aggregated_inputs.py)  
Pre-processing of tabular impacts data of exposure by GWL, into netcdf datasets that will be used in emulation. Only needs to run once to pre-process the impacts data. Only required if working with IAMC table impacts data.

### [`pp_combined example.ipynb`](https://github.com/iiasa/rime/blob/main/rime/pp_combined_example.py)  
Example jupyter notebook that demonstrates methods of processing both table and map impacts data for IAM scenarios.

or [
### [`process_tabledata.py`](https://github.com/iiasa/rime/blob/main/rime/process_tabledata.py)  
Example script that takes input table of emissions scenarios with global temperature timeseries, and output tables of climate impacts data in IAMC format. Can be done for multiple scenarios and indicators at a time. 
and
### [`process_maps.py`](https://github.com/iiasa/rime/blob/main/rime/process_tabledata.py)  
Example script that takes input table of emissions scenarios with global temperature timeseries, and output maps of climate impacts through time as netCDF. Ouptut netCDF can be specified for either for 1 scenario and multiple climate impacts, or multiple scenarios for 1 indicator.
]


### [`test_map_notebook.html`](https://github.com/iiasa/rime/blob/main/rime/test_map_notebook.html)
Example html maps dashboard. CLick download in the top right corner and open locally in your browser.

![image](https://github.com/iiasa/rime/assets/17701232/801e2dbe-cbe8-482f-be9b-1457c92ea23e)


## Installation

At command line, navigate to the directory where you want the installation, e.g. your Github folder.  

	git clone https://github.com/iiasa/rime.git

Change to the rime folder and install the package including the requirements.  

	pip install --editable .

## Further information
This package is in a pre-release mode, currently work in progress, under-going testing and not yet formally published.  

Examples provided use climate impacts data that is currently under-going peer-review [(Werning et al. 2024)](https://zenodo.org/records/10868066), currently hosted on the [Climate Solutions Explorer](https://www.climate-solutions-explorer.eu/).

