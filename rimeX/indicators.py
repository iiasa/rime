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

def rx5day(input_files, output_file, previous_input_files=None, previous_output_file=None, **kw):
    """ annual maximum of 5-day precipitation (in mm)
    """
    assert len(input_files) == 1
    seconds_per_day = 86400

    if previous_input_files is None:
        input = input_files[0]
    else:
        assert len(previous_input_files) == 1
        input = f"-cat -seltimestep,-4/-1 {previous_input_files[0]} {input_files[0]}"

    cdo(f"-mulc,{seconds_per_day} -yearmax -shifttime,+2days -runsum,5 {input} {output_file}", **kw)

    # Rename the variable 'pr' to 'rx5day' and overwrite the file
    check_call(f"ncrename -v pr,rx5day {output_file}", **kw)

    # Update the units and standard_name and long_name attributes for 'rx5day' and overwrite the file
    check_call(f"ncatted -O -a units,rx5day,o,c,'mm' {output_file}", **kw)
    check_call(f"ncatted -O -a standard_name,rx5day,o,c,'maximum_5_day_precipitation_amount' {output_file}", **kw)
    check_call(f"ncatted -O -a long_name,rx5day,o,c,'Maximum 5-day precipitation amount' {output_file}", **kw)