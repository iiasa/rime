# rimeX

## Description

This repository contains code originally written for the [Climate Impact Explorer](https://climate-impact-explorer.climateanalytics.org).
It started as a rewrite of the emulator intended for provide better statistical calculations with exact frequentist estimates.
It was moved to a standalone repository for re-use in various projects, and is intended to supercede the code for the [Rapid Impact Model Emulator](https://github.com/iiasa/rime) (hence its name).


## Back-compatibility and transition period

For users who want to use the original module by Edward Byers instead, the `rimeX.legacy` subpackage is made available.
All `rime` imports were updated with `rimeX.legacy`, but it is otherwise left unedited since import on March 22nd, 2024: `sed -i "s/rime\./rimeX.legacy./g" *.py wip_scraps/*.py`.

It is possible to import via `import rimeX.legacy as rime` to use existing code. Eventually this subpackage will be deprecated.


## Install

A development install can be done after cloning the repo, in pip-editable `-e` mode (that way code edits will propagate without the need for re-installing):

	git clone --single-branch --branch rimeX https://github.com/iiasa/rime.git
	cd rime
	pip install -e .

For the end-user (we're not at this stage yet) or one-off testing, 
it's also possible to do it in one go with pip, but the whole repo is cloned in the background so it's slower. 
The command is shown below for completeness, but it is not recommended (slower and no edits possible):

 	pip install git+https://github.com/iiasa/rime.git@rimeX


To install all optional dependencies, append `[all]`, e.g. from the local clone:

	pip install -e .[all]


## Usage

The following scripts are made available, for which inline help is available with `-h` or `--help`:

- Data download and pre-processing scripts (presently ISIMIP only, variables tas and pr)

	- `rime-download-isimip` : download ISIMIP data
	- `rime-download` : download other datasets (Werning et al 2024) etc. (platform-independent)
  	- `rime-pre-gmt` : pre-processing: crunch global-mean-temperature
	- `rime-pre-region` : pre-precessing: crunch regional averages (=> this currently requires Climate Impact Explorer masks)

- Actually use the emulator
	
	- `rime-init-wl` : crunch the warming levels (required)
	- `rime-init-digitize` : pre-compute digitized regional average based on warming levels (optional)
	- `rime-run-timeseries` : run the main emulator (time-series)

- Also useful:

	- `rime-config` : print the config to screen (toml format)

Of course, any of the functions can be called directly. Inline documentation is available. 


## Example Usage:

Below a simple example using [ixmp4](https://docs.ece.iiasa.ac.at/projects/ixmp4/en/latest/data-model.html) input files from AR6 WG3 scenarios with [Werning et al 2024](https://zenodo.org/records/6496232) datasets:

	$ rime-download --ls
	Available datasets are:
	  werning2024/table_output_avoided_impacts werning2024/table_output_climate_exposure werning2024/precipitation werning2024/temperature werning2024/air_pollution werning2024/energy werning2024/hydrology werning2024/land AR6-WG3-plots/spm-box1-fig1-warming-data.csv AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv

	$ rime-download --name AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv werning2024/table_output_climate_exposure

	$ rime-run-timeseries --iam-file AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv --iam-variable "*GSAT*median*" --iam-filter category_show_lhs=C6 --impact-file werning2024/table_output_climate_exposure/table_output_heatwave_COUNTRIES.csv --impact-region ITA --impact-variable "hw_95_10|Exposure|Population|%" -o output.csv --overwrite

The example above requires the filtering of exactly one time-series and one impact type from the multidimensional input files. It will issue an error message if more than one temperatrure scenario is present. This preliminary version only accounts from the warming level in the impact dataset. The years and ssp family are considered an "uncertainty" and they show up as quantiles in the output file (in this example they are the only contributor). More functionality will be added soon.


## Config files and default parameters

Note the scripts sets default parameters from a [configuration file](rimeX/config.toml).
You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. The `rime-config` script is provided to output the default config to standard output, to save it to file e.g. `rime-config > rime.toml` and later edit `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.

If used interactively or imported from a custom script, the config can be changed on-the-fly by accessing the `rimeX.config.CONFIG` flat dictionary.

By default, ISIMIP3b data are used, but that can be changed to ISIMIP2b via the `--simulation-round` flag (available models and experiments and defaults are adjusted automatically).


## TODO

- Provide test data
- Function to download pyiam scenarios 
- More options for data download
- More options for regional averages
	- options to download ISIMIP or CIE or other masks
- Support various backend formats for data (CSV time-series organized in folder, netCDF files etc)
- emulator functionality: add functionality while keeping the code clean --> consider adding class mechanism
