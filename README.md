# RIME - Rapid Impact Model Emulator

2023 IIASA

[![license](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)](https://choosealicense.com/licenses/gpl-3.0/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

[![RIME_logo]](https://github.com/iiasa/rime/assets/17701232/12e9ae66-5d28-4f06-9540-fa496cc588d0)  

## Overview  
------------------  

**RIME** is a lightweight software tool for using global warming level approaches to link climate impacts to Integrated Assessment Models (IAMs).

When accompanied by climate impacts data (table and/or maps), RIME can be used to take a global mean temperature timeseries (e.g. from an IAM or climate model like FaIR/MAGICC), and return tables and maps of climate impacts through time consistent with the warming of the scenario.  

There are two key use-cases for the RIME approach:  
1. Estimating a suite of climate impacts from an global emissions or temperature scenario.  
2. Reformulating climate impacts data to be used as an input to an integrated assessment model scenario.  






### `rime_functions.py` 
Contains the key functions that can be used to process data. 

### `utils.py`
A collection of helper functions related to data processing, used within functions and as standalone, if needed.

### `process-config.py` 
A script to host a large number of configurable settings for running the software on datasets.
Needs to be imported at the beginning of a script, e.g. `from process_config import *`.  
Settings for Dask could be configured in here. 

### `generate_aggregated_inputs.py` 
Pre-processing of tabular impacts data of exposure by GWL, into netcdf datasets that will be used in emulation. Only needs to run once to pre-process the impacts data. 

### `process_tabledata.py` 
Take input table of emissions scenarios with GMT and output tables of climate impacts data in IAMC format. Can be done for multiple scenarios and indicators at a time. 

### `process_maps.py`  
Take input table of emissions scenarios with GMT and output maps of climate impacts through time as netCDF. Ouptut netCDF can be specified for either for 1 scenario and multiple climate impacts, or multiple scenarios for 1 indicator.


## Tutorials
Example scripts and tutorials are found inthe respective folders.


## Installation

At command line, navigate to the directory where you want the installation, e.g. your Github folder.  

	`git clone https://github.com/iiasa/rime.git`

Change to the rime folder and install the package including the requirements.  

    pip install --editable .

