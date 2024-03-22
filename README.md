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

	git clone --single-branch --branch rimeX https://github.com/iiasa/rime.git
	cd rimeX
	pip install .

It's also possible to do it in one go with pip, but the whole repo is cloned in the background so it's slower.

 	pip install git+https://github.com/iiasa/rime.git@rimeX

To install all optional dependencies do instead (from the local clone):

	pip install .[all]


## Usage

The following scripts are made available, for which inline help is available with `-h` or `--help`:

- Data download and pre-processing scripts (presently ISIMIP only, variables tas and pr)

	- `rime-download-isimip` : download ISIMIP data
  	- `rime-pre-gmt` : pre-processing: crunch global-mean-temperature
	- `rime-pre-region` : pre-precessing: crunch regional averages (=> this currently requires Climate Impact Explorer masks)

- Actually use the emulator
	
	- `rime-init-wl` : crunch the warming levels (required)
	- `rime-init-digitize` : pre-compute digitized regional average based on warming levels (optional)
	- `rime-run-timeseries` : run the main emulator (time-series)

- Also useful:

	- `rime-config` : print the config to screen (toml format)

Of course, any of the functions can be called directly. Inline documentation is available. 


## Config files and default parameters

Note the scripts sets default parameters from a [configuration file](rimeX/config.toml), which is set to fetch ISIMIP3 data by default. 
You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. The `rime-config` script is provided to output the default config to standard output, to save it to file e.g. `rime-config > rime.toml` and later edit `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.


Note the config file also sets the defaults at the function level (that might be changed in the future).
In case the functions are imported directly from rimeX, it is possible to read from a custom config via `rimeX.set_config`. 
Note in case other modules have already been imported, they'll need to be reloaded for the defaults changes to take effect (see `importlib.reload`).


## TODO

- Provide test data
- Function to download pyiam scenarios 
- More options for data download
- More options for regional averages
	- options to download ISIMIP or CIE or other masks
- Support various backend formats for data (CSV time-series organized in folder, netCDF files etc)
- emulator functionality: add functionality while keeping the code clean --> consider adding class mechanism
