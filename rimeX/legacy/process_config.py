# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:28:48 2023

@author: byers

File within which to configure the settings and working directories
To be imported at start of each file run using 
from process_config import *

"""
# =============================================================================
# process_config.py
# =============================================================================
import os

import rimeX.legacy as oldrime
from rimeX.datasets import get_datapath
from rimeX.config import get_outputpath

# Run and environment settings

env = "pip"
# env = "pc"
# env = 'server'
# env = 'ebro3'



# From generate_aggregated_inputs.py

region = "COUNTRIES"  # 'R10' or 'COUNTRIES'
# region = "R10"
table_output_format = f"table_output_|_{region}.csv"


yr_start = 2010
yr_end = 2100


# Dask settings
num_workers = 24  # Number of workers. More workers creates more overhead
parallel = True  # Uses Dask in processing the IAMC scenarios


caution_checks = True


# =============================================================================
# %% Working directories
# =============================================================================

TEST_DATA = get_datapath("test_data")
yaml_path = os.path.join(oldrime.__path__[0], "indicator_params.yml")

if env != "pip":
    landmask_path = os.path.join(git_path, "climate_impacts_processing", "landareamaskmap0.nc")
    kg_class_path = os.path.join(git_path, "climate_impacts_processing", "kg_class.nc")
else:
    landmask_path = TEST_DATA
    kg_class_path = TEST_DATA

if env == "pc":
    # git_path = f"C:\\users\\{user}\\Github\\"    
    git_path = f"C:\\Github\\"    
    user = "byers"    
    # Main working directory
    wd = f"C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Research Theme - NEXUS\\Hotspots_Explorer_2p0\\"
    wd_input = f"P:\\watxene\\ISIMIP_postprocessed\\cse\\"  # Input data branch
    # Directory of table files to read as input
    wdtable_input = "table_output\\"

    # Output directory
    wd2 = "rcre_testing\\testing_3\\"
    output_dir = f"{wd}{wd2}aggregated_region_datafiles\\"

    # Input source of processed climate data by ssp/year/variable/region
    folder_input_climate = "aggregated_region_datafiles\\"
    fname_input_climate = f"{wd}{wd2}{folder_input_climate}*_{region}*.nc"

    # Input IAMC scenarios file, must have a temperature variable
    fname_input_scenarios = get_datapath("test_data/emissions_temp_AR6_small.xlsx")

    # Directory of map files to read as input
    impact_data_dir = f"{wd}\\data\\4_split_files_for_geoserver"
    # impact_data_dir = f"{wd_input}split_files"

else:
    # Main working directory
    # wd = f"C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Research Theme - NEXUS\\Hotspots_Explorer_2p0\\"
    # wd_input = f"P:\\watxene\\ISIMIP_postprocessed\\cse\\"  # Input data branch
    # Directory of table files to read as input
    # wdtable_input = "table_output\\"

    # Output directory
    # wd2 = "rcre_testing\\testing_3\\"
    # output_dir = f"{wd}{wd2}aggregated_region_datafiles\\"

    # Input source of processed climate data by ssp/year/variable/region
    # folder_input_climate = "aggregated_region_datafiles\\"
    fname_input_climate = str(TEST_DATA / f"*_{region}*.nc")

    # Input IAMC scenarios file, must have a temperature variable
    fname_input_scenarios = get_datapath("test_data/emissions_temp_AR6_small.xlsx")

    # Directory of map files to read as input
    impact_data_dir = get_datapath("werning2024")
    # impact_data_dir = f"{wd_input}split_files"

# =============================================================================
# %% From process-iamc_scenarios_gwl.py
# =============================================================================

year_resols = [5]
input_scenarios_name = "AR6full"

temp_variable = (
    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile"
)
ssp_meta_col = "Ssp_family"  # meta column name of SSP assignment

if env != "pip":
    output_folder_tables = f"{wd}{wd2}output\\tables\\"
    output_folder_maps = f"{wd}{wd2}output\\maps\\"
else:
    output_folder_tables = get_outputpath("tables")
    output_folder_maps = get_outputpath("maps")


prefix_indicator = "Climate impacts|RIME|"


few_scenarios = False
very_few_scenarios = False
# few_variables = True
testing = False
test = "" if testing == False else "para"
lvaris = 200


# %% =============================================================================
# from interpolate_maps.py
# =============================================================================

# impact data settings

indicators = ["cdd", "precip"]
ftype = "score"  # score
interpolation = 0.01


# scenario data settings
years = range(2015, 2101, 5)
scenarios = {"AIM/CGE 2.0": "SSP1-26", "GCAM 5.3": "SSP_SSP5"}
sspdic = {1.0: "ssp1", 2.0: "ssp2", 3.0: "ssp3", 4.0: "ssp4", 5.0: "ssp5"}


# =============================================================================
# %% Functions
# =============================================================================
