"""This module contains functions to calculate climate indicators

The function signature is:

    def indicator(input_files, output_file, previous_input_files=None, previous_output_file=None, dry_run=False):
        pass

where:
    input_files: list of input files
    output_file: output file
    previous_input_files: list of input files from the previous time step
    previous_output_file: file from the previous time step
    dry_run: if True, the function should only print the command to be executed
"""
from rimeX.tools import check_call, cdo


SECONDS_PER_DAY = 86400

def rx5day(input_files, output_file, previous_input_files=None, dry_run=False, **kw):
    """ annual maximum of 5-day precipitation (in mm)
    """
    assert len(input_files) == 1

    if previous_input_files is None:
        input = input_files[0]
    else:
        assert len(previous_input_files) == 1
        input = f"-cat -seltimestep,-4/-1 {previous_input_files[0]} {input_files[0]}"

    cdo(f"-mulc,{SECONDS_PER_DAY} -yearmax -shifttime,+2days -runsum,5 {input} {output_file}", dry_run=dry_run)

    # Rename the variable 'pr' to 'rx5day'
    check_call(f"ncrename -v pr,rx5day {output_file}", dry_run=dry_run)

    # Update the units and standard_name and long_name attributes for 'rx5day'
    check_call(f"ncatted -O -a units,rx5day,o,c,'mm' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a standard_name,rx5day,o,c,'maximum_5_day_precipitation_amount' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a long_name,rx5day,o,c,'Maximum 5-day precipitation amount' {output_file}", dry_run=dry_run)


def heavy_precipitation_days(input_files, output_file, **kw):
    """ annual number of days with more then 10mm of precipitation
    """
    import pandas as pd
    import xarray as xa
    import os
    

    assert len(input_files) == 1

    threshold = 1 / SECONDS_PER_DAY  # 1 mm/day in mm/s

    daily_precip = xa.open_dataset(input_files[0])
    
    heavy_precipitation_days = xa.where(daily_precip.pr >= 10 * threshold , 1, 0).groupby('time.year').sum('time').to_dataset().rename({'year':'time', 'pr': 'heavy_precipitation_days'})
    heavy_precipitation_days['time'] =  pd.to_datetime(heavy_precipitation_days['time'].values,format='%Y')

    heavy_precipitation_days.to_netcdf(output_file)


def very_heavy_precipitation_days(input_files, output_file, **kw):
    """ annual number of days with more then 20mm of precipitation
    """
    import pandas as pd
    import xarray as xa
    import os
    

    assert len(input_files) == 1

    threshold = 1 / SECONDS_PER_DAY  # 1 mm/day in mm/s

    daily_precip = xa.open_dataset(input_files[0])
    
    very_heavy_precipitation_days = xa.where(daily_precip.pr >= 20 * threshold , 1, 0).groupby('time.year').sum('time').to_dataset().rename({'year':'time', 'pr': 'very_heavy_precipitation_days'})
    very_heavy_precipitation_days['time'] =  pd.to_datetime(very_heavy_precipitation_days['time'].values,format='%Y')

    very_heavy_precipitation_days.to_netcdf(output_file)
    very_heavy_precipitation_days.close()
    daily_precip.close()
  
def wet_days_cse(input_files, climatology_files, output_file, **kw):
    '''Number of days above the 95th historical precipitation percentile'''
    import xarray as xa
    import pandas as pd
    

    climatology = xa.open_dataset(climatology_files[0]).squeeze()
    
    wet_days = xa.open_dataset(input_files[0])

    wet_days = xa.where(wet_days.pr >= climatology.pr, 1, 0).groupby('time.year').sum('time').to_dataset().rename({'year':'time', 'pr': 'wet_days_cse'})

    wet_days['time'] =  pd.to_datetime(wet_days['time'].values,format='%Y')

    wet_days.to_netcdf(output_file)

    climatology.close() 
    wet_days.close()

def very_wet_days_cse(input_files, climatology_files, output_file, **kw):
    '''Number of days above the 95th historical precipitation percentile'''
    import xarray as xa
    import pandas as pd

    climatology = xa.open_dataset(climatology_files[0])
    
    very_wet_days = xa.open_dataset(input_files[0])

    very_wet_days = xa.where(very_wet_days.pr >= climatology.pr, 1, 0).groupby('time.year').sum('time').to_dataset().rename({'year':'time', 'pr': 'very_wet_days_cse'})

    very_wet_days['time'] =  pd.to_datetime(very_wet_days['time'].values,format='%Y')
    
    very_wet_days.to_netcdf(output_file)

    climatology.close()
    very_wet_days.close()

def annual_drought_intensity(input_files, climatology_files, output_file, **kw):

    import xarray as xa
    import pandas as pd
    import os

    quantiles = xa.open_dataset(climatology_files[0]).squeeze().quantile(0.1, dim='time',skipna = True)
    
    discharge = xa.open_dataset(input_files[0])

    
    duration = discharge.where(discharge < quantiles)
    duration_annual_sum = duration.groupby('time.year').count(dim='time')

    deficit = (quantiles - discharge).where(quantiles > discharge)
    deficit_annual_sum = deficit.groupby('time.year').sum(dim='time')

    drought_intensity_annual = deficit_annual_sum / duration_annual_sum

    out = xa.merge([drought_intensity_annual.rename({'qtot': 'annual_drought_intensity', 'year':'time'})]).fillna(0)
    out['time'] =  pd.to_datetime(out['time'].values,format='%Y')
    
    out.to_netcdf(output_file)

    os.remove(input_files[0]) 
    print(f'Finished processing and removed {input_files[0]}')

    out.close()
    deficit.close()
    deficit_annual_sum.close()
    duration.close()
    duration_annual_sum.close()
    quantiles.close()
    duration.close()

def cooling_degree_days(input_files, output_file, **kw):
    import xarray as xr
    import pandas as pd
    import os

    
    TAS = xr.open_dataset(input_files[0])

    
    simple_degree_days_cooling =  TAS.where(TAS.tas > 273.15+26) - (273.15 + 26)
    
    simple_degree_days_yearly = simple_degree_days_cooling.groupby('time.year').sum(dim='time', skipna=True)

    simple_degree_days_yearly = simple_degree_days_yearly.rename({'tas': 'cooling_degree_days', 'year':'time'})

    simple_degree_days_yearly['time'] =  pd.to_datetime(simple_degree_days_yearly['time'].values,format='%Y')
    
    simple_degree_days_yearly.to_netcdf(output_file)

    os.remove(input_files[0]) 
    print(f'Finished processing and removed {input_files[0]}')
    TAS.close()
    simple_degree_days_yearly.close()
    simple_degree_days_cooling.close()




def simple_precipitation_intensity_index(input_files, output_file, **kw):
    '''Number of days above the 95th historical precipitation percentile'''
    import xarray as xa
    import pandas as pd

    
    threshold = 1 / SECONDS_PER_DAY  # 1 mm/day in mm/s

    daily_precip = xa.open_dataset(input_files[0])
    
    precipitation_days = xa.where(daily_precip.pr >= 1 * threshold , 1, 0).groupby('time.year').sum('time').rename({'year':'time'})
    precipitation_days['time'] =  pd.to_datetime(precipitation_days['time'].values,format='%Y')

    precip_sum = daily_precip.groupby('time.year').sum('time').rename({'year':'time'})
    precip_sum['time'] =  pd.to_datetime(precip_sum['time'].values,format='%Y')

    simple_precipitation_intensity_index = precip_sum / precipitation_days

    simple_precipitation_intensity_index = simple_precipitation_intensity_index.rename({'pr': 'simple_precipitation_intensity_index'})

    print(simple_precipitation_intensity_index.time)



def daily_temperature_variability(input_files, climatology_files, output_file, dry_run=False, **kw):
    """ daily temperature variability (in deg. C) after Kotz et al. (2024) (Eq. 1)
    """
    assert len(input_files) == 1
    assert len(climatology_files) == 1

    cdo(f"yearmean -sqrt -monmean -pow,2 -ymonsub {input_files[0]} {climatology_files[0]} {output_file}", dry_run=dry_run)

    # Rename the variable 'tas' to 'daily_temperature_variability'
    name = "daily_temperature_variability"
    check_call(f"ncrename -v tas,{name} {output_file}", dry_run=dry_run)

    # Update the units and standard_name and long_name attributes for 'daily_temperature_variability'
    check_call(f"ncatted -O -a units,{name},o,c,'degrees_celsius' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a standard_name,{name},o,c,'daily_temperature_variability' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a long_name,{name},o,c,'Daily temperature variability' {output_file}", dry_run=dry_run)


def number_of_wet_days(input_files, output_file, dry_run=False, **kw):
    """ Number of wet days after Kotz et al. (2024) (Eq. 2)
    """
    assert len(input_files) == 1

    threshold = 1 / SECONDS_PER_DAY  # 1 mm/day in mm/s

    cdo(f"yearsum -gtc,{threshold} {input_files[0]} {output_file}", dry_run=dry_run)

    name = "number_of_wet_days"
    check_call(f"ncrename -v pr,{name} {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a units,{name},o,c,'days' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a standard_name,{name},o,c,'number_of_wet_days' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a long_name,{name},o,c,'Number of wet days' {output_file}", dry_run=dry_run)


def extreme_daily_rainfall(input_files, climatology_files, output_file, dry_run=False, **kw):
    """ Extreme daily rainfall (in mm) after Kotz et al. (2024) (Eq. 3)
    """
    assert len(input_files) == 1
    assert len(climatology_files) == 1

    cdo(f"-mulc,{SECONDS_PER_DAY} -yearsum -mul {input_files[0]} -gt {input_files[0]} {climatology_files[0]} {output_file}", dry_run=dry_run)

    # Rename the variable 'tas' to 'extreme_daily_rainfall'
    name = "extreme_daily_rainfall"
    check_call(f"ncrename -v pr,{name} {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a units,{name},o,c,'mm' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a standard_name,{name},o,c,'extreme_daily_rainfall' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a long_name,{name},o,c,'Extreme daily rainfall' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a desc,{name},o,c,'Annual precipitation above the historical 99.9th percentile' {output_file}", dry_run=dry_run)