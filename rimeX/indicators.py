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

    cdo(f"-mulc,{SECONDS_PER_DAY} yearsum -mul {input_files[0]} -gt {input_files[0]} {climatology_files[0]} {output_file}", dry_run=dry_run)

    # Rename the variable 'tas' to 'extreme_daily_rainfall'
    name = "extreme_daily_rainfall"
    check_call(f"ncrename -v pr,{name} {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a units,{name},o,c,'mm' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a standard_name,{name},o,c,'extreme_daily_rainfall' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a long_name,{name},o,c,'Extreme daily rainfall' {output_file}", dry_run=dry_run)
    check_call(f"ncatted -O -a desc,{name},o,c,'Annual precipitation above the historical 99.9th percentile' {output_file}", dry_run=dry_run)