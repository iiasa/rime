# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:28:48 2023

@author: byers

File within which to configure the settings and working directories
To be imported at start of each file run

"""
# =============================================================================
# process_config.py
# =============================================================================
import os

# Run and environment settings
user = "byers"
env = "pc"
# env = 'server'
# env = 'ebro3'

# git_path = f"C:\\users\\{user}\\Github\\"
git_path = f"C:\\Github\\"



# From generate_aggregated_inputs.py

region = "COUNTRIES"  # 'R10' or 'COUNTRIES'
table_output_format = f"table_output_|_{region}.csv"


yr_start = 2010
yr_end = 2100



num_workers = 24
parallel = True  # Uses Dask in processing the IAMC scenarios


caution_checks = True


# =============================================================================
# %% Working directories
# =============================================================================


yaml_path = git_path+"climate_impacts_processing\\hotspots.yml"
landmask_path = git_path+"climate_impacts_processing\\landareamaskmap0.nc"
kg_class_path = git_path+"climate_impacts_processing\\kg_class.nc"

if env == "pc":
    # Main working directory
    wd = f"C:\\Users\\{user}\\IIASA\\ECE.prog - Documents\\Research Theme - NEXUS\\Hotspots_Explorer_2p0\\"
    # Directory of table files to read as input
    wdtable_input = f"{wd}data\\5_table_outputs\\"
    # Output directory
    wd2 = "rcre_testing\\testing_2\\"
    output_dir = f"{wd}{wd2}aggregated_region_datafiles\\"

    # Input source of processed climate data by ssp/year/variable/region
    folder_input_climate = "aggregated_region_datafiles\\"
    fname_input_climate = f"{wd}{wd2}{folder_input_climate}*_{region}*.nc"

    # Input IAMC scenarios file, must have a temperature variable
    fname_input_scenarios = f"{wd}{wd2}emissions_temp_AR6_small.xlsx"

    # Directory of map files to read as input
    impact_data_dir = f"{wd}\\data\\4_split_files_for_geoserver"


elif env == "ebro3":
    wd = "H:\\"
    wd2 = "H:\\rcre_testing\\"

    # Input source of processed climate data by ssp/year/variable/region
    folder_input_climate = "aggregated_region_datafiles\\"
    fname_input_climate = f"{wd2}{folder_input_climate}*_{region}.nc"

    # Input IAMC scenarios file, must have a temperature variable
    fname_input_scenarios = f"{wd2}emissions_temp_AR6_small.xlsx"

    output_dir = f"{wd2}aggregated_region_datafiles\\"

    impact_data_dir = f"{wd}\\data\\4_split_files_for_geoserver"


# =============================================================================
# %% From process-iamc_scenarios_gmt.py
# =============================================================================

year_resols = [5]
input_scenarios_name = "AR6full"

temp_variable = (
    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile"
)
ssp_meta_col = "Ssp_family"  # meta column name of SSP assignment


output_folder_tables = f"{wd}{wd2}output\\tables\\"
output_folder_maps = f"{wd}{wd2}output\\maps\\"

prefix_indicator = "RCRE|"


few_scenarios = False
very_few_scenarios = False
# few_variables = True
testing = True
test = "" if testing == False else "para"
lvaris = 200


# %% =============================================================================
# from interpolate_maps.py
# =============================================================================

# impact data settings

indicators = ["cdd", "precip"]
ftype = "score" #score
interpolation = 0.01


# scenario data settings
years = range(2015, 2101, 10)
scenarios = {"AIM/CGE 2.0": "SSP1-26", "GCAM 5.3": "SSP_SSP5"}
sspdic = {1.0: "ssp1", 2.0: "ssp2", 3.0: "ssp3", 4.0: "ssp4", 5.0: "ssp5"}


# =============================================================================
# %% Functions
# =============================================================================
