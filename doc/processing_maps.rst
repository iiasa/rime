
process_maps.py
===============



Overview
--------

This example script takes an input table of emissions scenarios along with global temperature time series (`emissions_temp_AR6_small.xlsx`), and input gridded climate impacts data by global warming levels (e.g. `ISIMIP2b_dri_qtot_ssp2_2p0_abs.nc`) and generates maps of climate impacts over time as NetCDF files. It exemplifies the application of the RIME framework to spatially resolved climate impact data, remapping climate impacts data by global warming level to a trajectory of global mean temperature.

Usage
-----

The script's flexibility allows for the specification of outputs either for a single scenario across multiple climate impacts or for multiple scenarios focused on a single indicator. 


By processing emissions scenarios and associated temperature projections, ``process_maps.py`` produces NetCDF files that map climate impacts over time.
