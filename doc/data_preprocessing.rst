Pre-processing input table data
*********************

To work with table data, some pre-processing is likely required to achieve the correct formats. 

The aim is to go from typically tabular or database data, into a compressed 4-D netCDF format that is used in the emulation. For a given climate impacts dataset, this pre-processing only needs to be done once for preparation, and only if working with table data. Depending on the input dataset size, this can take some time.

The output netCDF has the dimensions:
	"gwl": for the global warming levels, at which impacts are calculated. (float)
	"year": for the year to which the gmt corresponds, if relevant, for example relating to exposure of a population of land cover in year x.
	"ssp": for the Shared Socioeconomic Pathway, SSP1, SSP2, SSP3, SSP4, SSP5. (str)
	"region": for the spatial region for the impact relates and might be aggregated to, e.g. country, river basin, region. (str)
	
	
Thus, the input data table should also have these dimensions, normally as columns, and additionally one for `variable`.

[example picture of IAMC input file]

The script `generate_aggregated_inputs.py` gives an example of this workflow, using a climate impacts dataset in table form (IAMC-wide), and converting it into a netCDF, primarily using the function `loop_inteprolate_gwl()`. In this case the data also has the `model` and `scenario` columns, which are not needed in the output dataset.

generate_aggregated_inputs.py
=============================


Key Features
------------

- **Data Aggregation**: Combines data from multiple files or data streams.
- **File Operations**: Utilizes glob and os modules for file system operations, indicating manipulation of file paths and directories.
- **Data Processing**: Imports ``xarray`` for working with multi-dimensional arrays, and ``pyam`` for integrated assessment modeling frameworks, suggesting complex data manipulation and analysis.

Dependencies
------------

- ``alive_progress``: For displaying progress bars in terminal.
- ``glob``: For file path pattern matching.
- ``os``: For interacting with the operating system's file system.
- ``pyam``: For analysis and visualization of integrated assessment models.
- ``re``: For regular expression matching, indicating text processing.
- ``xarray``: For working with labeled multi-dimensional arrays.
- ``time``: For timing operations.

Usage
-----

Based on the test data, the intention here is to read in a file like `table_output_cdd_R10.xlsx` and output a file that looks like `cdd_R10.nc`

