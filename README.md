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

- Data download and pre-processing scripts (presently ISIMIP only, variables tas and pr, written for the CIE dataset and masks)

	- `rime-download-isimip` : download ISIMIP data
	- `rime-download` : download other datasets (Werning et al 2024) etc. (platform-independent)
  	- `rime-pre-gmt` : pre-processing: crunch global-mean-temperature
	- `rime-pre-region` : pre-precessing: crunch regional averages (=> this currently requires Climate Impact Explorer masks)
	- `rime-pre-wl` : crunch the warming levels
	- `rime-pre-digitize` : pre-compute digitized regional average based on warming levels (optional)

- Actually use the emulator (works anywhere as long as the data is available)
	
	- `rime-run-timeseries` : run the main emulator with proper uncertainty calculations (time-series)
	- `rime-run-table` : vectorized version of `rime-run-timeseries` with on-the-fly interpolation, without uncertainties recombination
	- `rime-run-map` : run the map emulator

- Also useful to specify the data paths:

	- `rime-config` : print the config to screen (toml format)

Of course, any of the functions can be called directly. Inline documentation is available. 

See the associated [notebook](notebooks/readme.ipynb) to find the code to produce some of the figures below.


## Fast table emulator

We provide a faster emulator `rime-run-table` that is a straightforward interpolation of the input impact data.

	rime-run-table --nc-impact test_data/cdd_R10.nc test_data/pr_r10_R10.nc --gsat-file test_data/emissions_temp_AR6_small.xlsx --gsat-variable "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile" --gsat-filter model=IMAGE* scenario=SSP1-26 scenario=SSP1-45 --gsat-filter IMP_marker=ModAct IMP_marker=SP IMP_marker=GS IMP_marker=Neg -o table.csv

And the output is in CSV format by default:

	year,ssp_family,warming_level,gsat_model,gsat_scenario,variable,region,value
	2015,1.0,1.115556885487447,IMAGE 3.0.1,SSP1-26,cdd|Exposure|Land area,Countries of Latin America and the Caribbean,
	2015,1.0,1.115556885487447,IMAGE 3.0.1,SSP1-26,cdd|Exposure|Land area,Countries of South Asia; primarily India,	
	...

In the example above, we make use of advanced filtering, where each occurrence of `--gsat-filter` acts as a join between groups of data. The figure below show the results for the `pr_r10|Exposure|Population|%` variable and `Countries of South Asia; primarily India` area.

![](notebooks/images/table-interp-linear-join.png)

Internally, the impact data is transformed into a multi-dimensional `xarray.DataArray` with main dimensions `(warming_level, [year,])`, and broadcast dimensions `(scenario, variable, region, model)` and interpolated along the temperature pathway with `scipy.interpolate.RegularGridInterpolator`. By default, if the `year` and `ssp_family` (or `scenario`) are present in the impact data and in the temperature data, these will be matched with the temperature forcing as well. See also `--ignore-ssp` and `--ignore-year`. 

For more advanced usage, the underlying `rimeX.emulator.ImpactDataInterpolator` class is made available. The above would be achieved as follow:

	import xarray as xa
	import pyam
	from rimeX.emulator import ImpactDataInterpolator
	from rimeX.datasets import get_datapath

	gsat = pyam.IamDataFrame(get_datapath("test_data/emissions_temp_AR6_small.xlsx").filter(variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile")

	gsat = pyam.concat([
		gsat.filter(scenarios=["SSP1-26", "SSP1-45"], model="IMAGE*"),
		gsat.filter(IMP_markers=["ModAct", "SP", "GS", "Neg"]) 
		])

	impacts = xa.open_mfdataset([ 
		get_datapath("test_data/cdd_R10.nc"), 
		get_datapath("test_data/pr_r10_R10.nc") ])

	idi = ImpactDataInterpolator(impacts)

	results = idi(gsat.rename({"Ssp_family": "ssp_family"}, axis=1))

And for plotting the results, e.g.:

	variable = "pr_r10|Exposure|Population|%"
	region = "Countries of South Asia; primarily India"

	results[(results.variable == variable) & (results.region == region)].pivot(index='year', values='value', columns=['gsat_scenario', 'gsat_model']).sort_index(axis=1).plot(ax=ax)


## Maps

A map emulator is also available (see data download section below to obtain the data):

	$ rime-run-map --gsat-file AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv --gsat-filter category_show_lhs=C8 quantile=0.5 -i "werning2024/*/cse_cdd_ssp2_*_abs.nc4" -v cdd --gwl-dim threshold -o maps.nc --year 2020 2050 2070 -O --bbox -10 20 35 50

Here the input file is obtained via `rime-download --name werning2024/precipitation` and the warming levels are spread across various files with the `threshold` dimension indicating the warming levels, and the variable is named `cdd`. It would also be possible to indicate a list of file via `-v file1 file2` and give the warming levels explicitly via `--gwl-values 1.2 1.5` etc.

![](notebooks/images/maps.png)


Note only the warming levels are considered for interpolation. It is up to the user to match the correct SSP scenario and year, if necessary, and possibly compute one year at a time if each year and impact map is mapped on a specific SSP scenario and population trajectory. No class is made available here as this can be realized with the xarray package, with the `DataArray.interp` method.


## Data download

A selection of datasets is made available for easy download:

	$ rime-download --ls
	Available datasets are:
	  werning2024/table_output_avoided_impacts werning2024/table_output_climate_exposure werning2024/precipitation werning2024/temperature werning2024/air_pollution werning2024/energy werning2024/hydrology werning2024/land AR6-WG3-plots/spm-box1-fig1-warming-data.csv AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv

Below a simple example using [ixmp4](https://docs.ece.iiasa.ac.at/projects/ixmp4/en/latest/data-model.html) input files from AR6 WG3 scenarios with [Werning et al 2024](https://zenodo.org/records/6496232) datasets:

	$ rime-download --name AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv werning2024/table_output_climate_exposure


## Taking uncertainties into account

The `rime-run-timeseries` script is similar to but more general than `rime-run-table`, and it is designed to account for the full uncertainty. 
It operates on list of records that are grouped in various ways, rather than on pandas DataFrames or multi-dimensional arrays.
At the moment it accepts a single impact indicator and temperature pathway. Any remaining dimension will contribute to the uncertainty 
estimate. More in-depth description of the underlying assumption will be provided in Schwind et al (2024, in prep).

In essence, the main difference between `run-timeseries` and `run-table` is that the latter is structured (DataFrame -> DataArray -> ND numpy index) and the former is unstructured (list of records with `groupby`). The structured version can lead to speed for certain data forms, and allows on-the-fly interpolation without any preliminary data transformation (this is the main speed-up gain). See also the `--vectorize` option with [vectorize][#vectorize].

A few practical differences, that will be examplified below:

- only a single variable can be calculated at a time

- no assumption is made about which specific dimension should be used for indexing. Some fields have a specific meaning attached, such as `value`, `warming_level`, `year`, `scenario`, but by default all fields besides `value` are used for indexing (for grouping operation such as interpolation across warming levels). If some secondary dimensions vary along with values, such as `year` (or `midyear`), it is necessary to explicitly exclude then from the index via the `--pool` command, e.g. `--pool midyear`, or pass the index directly `--index model scenario`, say. Note that `warming_level` will always be added to the index.

- the binning and recombination method does not (yet?) allow for on-the-fly interpolation, so to interpolate across warming levels it is necessary to pass explicit parameters `--interp-warming-levels`, and in case `--match-year` is also specified, `--interp-year`.


### Comparison with `rime-run-table`

Here we run a systematic comparison between `rime-run-timeseries` and `rime-run-table`, using `AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv` dataset for temperature and impact file `isimip_binned_data/tas/ITA/ITA/latWeight/tas_ita_ita_annual_latweight_21-yrs.csv`.

First let's define the common part to all commands below:

	COMMON="--gsat-file AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv --gsat-filter category_show_lhs=C6 --impact-file isimip_binned_data/tas/ITA/ITA/latWeight/tas_ita_ita_annual_latweight_21-yrs.csv"


- Median GSAT and single impact model and scenario:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585  -o rimeX_ITA_median_one_ts.csv

		rime-run-table $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585  -o rimeX_ITA_median_one_table.csv --pool midyear --ignore-ssp

- Median GSAT and all impact models and scenarios:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 -o rimeX_ITA_median_all_ts.csv
	
		rime-run-table $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 -o rimeX_ITA_median_all_table.csv  --ignore-ssp

- Resampled GSAT and single impact model and scenario:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585 -o rimeX_ITA_resampled_one_ts.csv
	
		rime-run-table $COMMON --gsat-variable "*GSAT*" --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585 -o rimeX_ITA_quantile_one_table.csv --ignore-ssp

- Resampled GSAT and all impact model and scenario:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 -o rimeX_ITA_resampled_all_ts.csv
	
		rime-run-table $COMMON --gsat-variable "*GSAT*" --gsat-filter category_show_lhs=C6 -o rimeX_ITA_quantile_all_table.csv --ignore-ssp


And here the results:

![](notebooks/images/comparison_table.png)


Now let's see what happen when we interpolate the impact results to have finer warming levels:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585  -o rimeX_ITA_median_one_ts.csv --pool midyear --interp-warming-levels
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 -o rimeX_ITA_median_all_ts.csv --pool midyear --interp-warming-levels
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585 -o rimeX_ITA_resampled_one_ts.csv --pool midyear --interp-warming-levels
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 -o rimeX_ITA_resampled_all_ts.csv --pool midyear --interp-warming-levels


![](notebooks/images/comparison_table_interp.png)


Finally, we compare with the `vectorized` version (see [vectorize](#vectorize)), which involves deterministic resampling of the impact data and temperature forcing, and can also use scipy interpolator to bypass the need for interpolating records:

		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585  -o rimeX_ITA_median_one_ts.csv --vectorize --sample 1000
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 -o rimeX_ITA_median_all_ts.csv --vectorize --sample 1000
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 --impact-filter model=CNRM-CM6-1 scenario=ssp585 -o rimeX_ITA_resampled_one_ts.csv --vectorize --sample 1000
		rime-run-timeseries $COMMON --gsat-variable "*GSAT*" --gsat-resample --gsat-filter category_show_lhs=C6 -o rimeX_ITA_resampled_all_ts.csv --vectorize --sample 1000


![](notebooks/images/comparison_table_vectorized.png)


Some comments:

- the `rime-run-table` script requires `--ignore-ssp` so that it does not attempt to match GSAT `ssp-family` with the impact table's SSP family (this must be done on-request in `rime-run-timeseries`)

- the `rime-run-timeseries` script uses `--interp-warming-levels` to interpolate the impact table prior to binning (default step of 0.01 deg. C), for a smoother result (this is done by default in `rime-run-table` since it relies on `scipy.interpolate.RegularGridInterpolator`).

- The resamples GSAT (almost) perfectly covers the `table` version, as desired (especially with interpolation)

- The `rime-run-table` version (single, dashed color lines) is an outer product of all GSAT (whichever quantiles are present) and `impacts`. Each line represents a different combination of impact `model, scenario` pair and `gsat_quantile`. In contrast the `rune-run-timeseries` version combines those into bins.

- The `--vectorize` flag involves resampling. By default 1000 samples are used. Since the resampling is deterministic (the data is resampled according to a weighted quantiles method), few samples are enough. (Randomness only plays a role when shuffling GSAT forcing data after resampling.)


See also the [todos](#todo) below.


### Match years

The defaults for `rime-run-timeseries` slightly differ from `rime-run-table`, but it can be brought to do similar things.

Let's come back to the Werning et al dataset:

	$ rime-run-timeseries --gsat-file AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv --gsat-variable "*GSAT*median*" --gsat-filter category_show_lhs=C6 --impact-file werning2024/table_output_climate_exposure/table_output_heatwave_COUNTRIES.csv --region MEX --variable "hw_95_10|Exposure|Population" --match-year-population --interp-years -o output.csv

Note that by default, the years and ssp family are considered an "uncertainty" and they show up as quantiles in the output file (in this example they are the only contributor). To match the year according to the GSAT time-series year, we add the `--match-year-population` option, 
and this requires prior interpolation of the years via `--interp-years`

![](notebooks/images/population_exposed_match_year.png)


### Fitting a distribution to quantiles (experimental)

If 5th and 95th percentiles are provided for GSAT in addition to the median, an underlying distribution can be inferred and the temperature resampled to obtain an extended error assessment. E.g. the above example can be modified by filtering for variable names `*GSAT*` (instead of `*GSAT*median*`) and adding `--gsat-resample`:

	 $ rime-run-timeseries [...] --region ITA --variable "hw_95_10|Exposure|Population|%" --gsat-variable "*GSAT*" --gsat-resample

![](notebooks/images/fit_and_resample.png)


Similarly, if quantiles are present, the impacts can be sampled with the flags `--impact-resample`. E.g.

	 $ rime-run-timeseries [...] --impact-file test_data/table_output_wsi_R10_pop_scaled_including_uncertainty.csv --variable "wsi|Exposure|Population" "wsi|Exposure|Population|5th percentile" "wsi|Exposure|Population|95th percentile" --impact-resample

Note how several impact variables can be specified so that three variables end up considered (corresponding to median, 5th and 95th percentiles). 
They are then sorted out by parsing the variable name (e.g. " 5th" or "|5th" is expected, and the median is whatever is left).

GSAT and impact distribution fitting can be combined (see figure below).

![](notebooks/images/fit_and_resample_gsat_and_impacts.png)


### Warming level steps

The default step for warming level interpolation is 0.1 degC. This is fine for a probabilistic setting, but sometimes it is preferrable to have finer warming level steps, especially when working with the median temperature time-series, to avoid visible aliasing. The option `--interp-warming-levels` with the default `--warming-level-step 0.01` is available (so far only available with the table format as input):

	 $ rime-run-timeseries [...] --interp-warming-levels --warming-level-step 0.01

Or preferably used the [vectorized](#vectorize) form:

	 $ rime-run-timeseries [...] --vectorize


![](notebooks/images/warming_level_step.png)


### Time step

The time-step is normally set by the input GSAT data file, but it can be subsampled or interpolated using `--time-step` (in years).

	 $ rime-run-timeseries [...] --time-step 5


![](notebooks/images/time_step.png)


### Vectorize

Vectorization uses `rimeX.emulator.recombine_gmt_vectorized`. This is a different method from the default `rimeX.emulator.recombine_gmt_ensembe` . The main is trick it does is to pack the impact records in a table form by resampling them, which involves quantile-interpolation and can be represented like this:

![](notebooks/images/vectorized_imapcts.png)

and then uses `RegularGridInterpolator` to combine with (also resampled) GMT.

Here a matrix of samples is returned, instead of quantiles, thus allowing probabilistic uses (with 100 samples below):

![](notebooks/images/probabilistic_forecast.png)

Note for some applications the result may need to be reshuffled, because the sample number is correlated to the impact, by construction.

I checked that with 1000 samples it is faster than first interpolating the records and then calling `recombine_gmt_ensemble`, and also smoother.

A hand-on example is provided in the readme notebook.

Probably in the future the vectorized form will be used as default with `rime-run-timseries`.


### Python API

The python API is currently unstable and will be documented in the future.

For now, the inner working consists in a set of functions to transform a list of records. By records, I mean dictionaries as the result of `pandas.DataFrame.to_dict('records')`, where each dict record corresponds to a row in a long pandas DataFrame. 
There are functions to interpolate the records w.r.t warming levels or years, average across scenarios, etc. These functions can be found in the `rimeX.records` module. The emulator itself, to combine GSAT and impact data, is present in the `rimeX.emulator` module (`rime-run-timeseries` relies on `recombine_gmt_ensemble` or `recombine_gmt_vectorized`, whereas `rime-run-table` relies on `recombine_gmt_table`).

Note the scripts are located in `rimeX.scripts` and can also be run in the notebook via module import for easier debugging (note the `--` separator to pass arguments to the module's `main()`, and not to `%run`):
- `rime-run-timeseries` as `%run -m rimeX.scripts.timeseries --` 
- `rime-run-table` as `%run -m rimeX.scripts.table --` 

Check out [the TODO internal classes](#internal-classes) section for an insight where it might be going.


## Config files and default parameters

Note the scripts sets default parameters from a [configuration file](rimeX/config.toml).
You can specify your own defaults by having a `rimeX.toml` or `rime.toml` file in the working directory (from which any of the above scripts are called), or by specifying any file via the command-line argument `--config <FILE.toml>`. The `rime-config` script is provided to output the default config to standard output, to save it to file e.g. `rime-config > rime.toml` and later edit `rime.toml` for custom use. Note it is OK to only define a few fields in the config file -- all others will be take from the default config.

If used interactively or imported from a custom script, the config can be changed on-the-fly by accessing the `rimeX.config.CONFIG` flat dictionary.

By default, ISIMIP3b data are used, but that can be changed to ISIMIP2b via the `--simulation-round` flag (available models and experiments and defaults are adjusted automatically).


## TODO

### Internal classes (maybe ...)

A prospective API would be a set of classes with methods, some of which would be shared across `rime-run-timeseries` and `rime-run-table`. The proposal below has a focus on the internal data structure.

- `ImpactRecords` -> internal data structure is a list of records: this is the base structure of `rime-run-timeseries`. The methods below return another `ImpactRecords` instance, unless otherwise specified:
	- interpolate_years
	- interpolate_warming_levels
	- mean : (current average_per_group)
	- make_equiprobable_groups
	- resample_from_quantiles 
	- resample_dims : resample from an index (e.g. model, scenario)  
	- resample_montecarlo : return an `ImpactEnsemble` instance
	- to_frame() (internally `ImpactFrame(pandas.DataFrame(self.records))`) would return an `ImpactFrame`
	- sample_by_gmt_pathway() -> return a DataFrame of results (current `recombine_gmt_ensemble`)

- `ImpactEnsemble` : DataFrame with an index (years as multi-index if other dimensions should be accounted for), and samples as columns, for vectorized sampling. 
	- sample_by_gmt_pathway() -> return a DataFrame of results (current `recombine_gmt_vectorized`)

- `ImpactFrame`: semantically equivalent to `ImpactRecords`, but internal data structure is a DataFrame : good for reading, writing and as an intermediate state, but operations of destruction / reconstruction can be costly. Eventually, this could be merged with `ImpactRecords`, with one or ther other taking over depending on performance tests. 
For now `ImpactRecords` is the main class for work, and `ImpactFrame` is mostly a data holder.
	- ... : some of the methods above may also be implemented using pandas methods (should be equivalent to ImpactRecords methods: ideal for unit tests)
	- to_cube(dims) -> transform to `ImpactCube`

- `ImpactCube` : current `ImpactDataInterpolator`, whose internal data structure is a DataArray
	- sel : (select e.g. specific SSP family)
	- interpolate_by_warming_levels(warming_levels)
	- interpolate_by_warming_levels_and_year()
	- to_frame() -> back to ImpactFrame, which can be more suitable for certain operations


### Scripts

In general: harmonize `rime-run-timeseries` and `rime-run-table`. The former should be able to do much of what the latter can do (except on-the-fly interp):

- `rime-run-table`: only match SSP and `year` on-demand
- `rime-run-timeseries`: add do not mix everything by default: use groupby (and mix on demand)
- both: pool [+ mean or other stat] scenario / years / models before interp


NOTE about `rime-run-table` ssp-family indexing:

- currently we pool SSP_family in impact data (mapped from scenarios), but for the above example there is not reason to do that

Currently we assume a set of "coordinates" dimensions to keep track of (`model, scenario, quantile, variable, year, warming_level, ...`) and to be passed to groupby. It's probably best to pass an `--index` parameter to specify what dimensions should be considered for indexing, groupby etc. : WE NOW DO THIS FOR `rime-run-timeseries`.

We currently "guess" some fields (lower case, remove space and hyphen, and even rename a few). Possibly use an explicit mapping as user input for renaming to standard fields without the current guessing.


### ADD UNIT TESTS
### Rename rimeX to rime
