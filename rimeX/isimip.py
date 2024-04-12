import os
import re
from itertools import product
from pathlib import Path
import argparse

from rimeX.tools import cdo
from rimeX.config import CONFIG, config_parser
from rimeX.logs import log_parser


client = None

def _init_client():
    global client
    if client is None:
        from isimip_client.client import ISIMIPClient
        client = ISIMIPClient()
    return client

def get_models(simulation_round=None):
    if simulation_round is None: simulation_round = CONFIG['isimip.simulation_round']
    return CONFIG["isimip"][simulation_round]["models"]

def get_experiments(simulation_round=None):
    if simulation_round is None: simulation_round = CONFIG['isimip.simulation_round']
    return CONFIG["isimip"][simulation_round]["experiments"]

def get_variables(simulation_round=None):
    return CONFIG["isimip.variables"]

def iterate_model_experiment(simulation_round=None):
    yield from product(get_models(simulation_round), get_experiments(simulation_round))

def _preparse_isimip_protocol():
    """First parse isimip protocol so that defaults models and experiments are correct
    """
    isimip_preparser = _get_isimip_parser()
    o, _ = isimip_preparser.parse_known_args()
    if o.simulation_round is not None:
        CONFIG["isimip.simulation_round"] = o.simulation_round

def _get_isimip_parser():
    """Determine if --simulation-round is provided in any script that relies on isimip, 
    so that get_models() and get_experiments() have the proper choices and default.
    The isimip_parser can be used in any 
    """
    isimip_parser = argparse.ArgumentParser(add_help=False)
    group = isimip_parser.add_argument_group('ISIMIP')
    group.add_argument("--experiment", nargs='+', default=get_experiments(), choices=get_experiments())
    _lower_to_upper = {m.lower():m for m in get_models()}
    group.add_argument("--model", nargs='+', default=get_models(), choices=get_models(), type=lambda s: _lower_to_upper.get(s, s))
    group.add_argument("--simulation-round", default=CONFIG["isimip.simulation_round"]) # already set in _prepare_isimip_protocol, but here for help

    return isimip_parser

_preparse_isimip_protocol()
isimip_parser = _get_isimip_parser()

_YEARS_ISIMIP3_RE = re.compile(r'(\d{4})_(\d{4}).nc')
_YEARS_ISIMIP2_RE = re.compile(r'(\d{4})\d{4}-(\d{4})\d{4}.nc')

def parse_years(path, simulation_round=None):
    if simulation_round is None: simulation_round = CONFIG['isimip.simulation_round']
    _YEARS_RE = _YEARS_ISIMIP2_RE if simulation_round is None or "ISIMIP2" in simulation_round else _YEARS_ISIMIP3_RE
    y1s, y2s = _YEARS_RE.search(Path(path).name).groups()
    y1, y2 = int(y1s), int(y2s)
    return y1, y2

def get_region_tag(bbox):
    l, b, r, t = bbox
    return f"lat{b}to{t}lon{l}to{r}"


def request_dataset(variables, experiment=None, model=None, download_folder='downloads', year_min=None, simulation_round=None):
    if year_min is None: year_min = CONFIG["isimip.historical_year_min"]
    if simulation_round is None: simulation_round = CONFIG['isimip.simulation_round']
    client = _init_client()
    results = []
    for v in variables:

        if model:
            iterable = ({"climate_variable": v, "climate_scenario":x, "climate_forcing":m.lower()} for m, x in product(model, experiment))
        elif experiment:
            iterable = ({"climate_variable": v, "climate_scenario":x} for x in experiment)
        else:
            iterable = ({"climate_variable": v}, )

        for kwargs in iterable:
            # response = client.datasets(simulation_round=simulation_round, bias_adjustment='w5e5', **kwargs)
            response = client.datasets(simulation_round=simulation_round, **kwargs)
            # response = client.datasets(simulation_round='ISIMIP3b', **kwargs)

            if response['count'] == 0:
                print(f"!!! No results found for {kwargs}")
            assert len(response["results"]) == response["count"]

            results.extend(response['results'])

    # filter historical files after 1980
    for r in results:
        r['files'] = [f for f in r['files'] if parse_years(f['path'], simulation_round)[0] >= year_min]

    return results


def main():
    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    parser.add_argument("-v", "--variable", nargs='+', choices=CONFIG["isimip.variables"])
        # default=['tas', 'pr', 'sfcwind', 'twet'])
    # parser.add_argument("--country", nargs='+', default=[], help='alpha-3 code or custom name from countries.toml')
    # parser.add_argument("--download-region", help='specify a region larger than --countries for the download')

    # These arguments come from an earlier, more general version of the code. They are not supposed to be changed here.
    # They are kept for back-compatibility but removed from the --help message to avoid confusion.
    group = parser.add_argument_group('Unused')
    group.add_argument("--country", nargs='+', default=[], help=argparse.SUPPRESS)
    group.add_argument("--daily", action='store_true', dest='daily', help=argparse.SUPPRESS)
    group.add_argument("--keep-daily", action='store_false', dest='remove_daily', help=argparse.SUPPRESS)
    group.add_argument("--mirror", help=argparse.SUPPRESS)  # in case we have direct access to PIK cluster, say
    group.add_argument("--download-folder", default=CONFIG["isimip.download_folder"], help=argparse.SUPPRESS)

    o = parser.parse_args()

    if not o.variable:
        print("At least one variable must be indicated. E.g. `--variable tas`")
        parser.exit(1)

    # import lower case to avoid any confusion in the outputs
    o.country = [c.lower() for c in o.country]

    # A previous version of the script allowed downloading for individual countries
    o.download_world = True

    # Request file list
    results = request_dataset(o.variable, experiment=o.experiment, model=o.model, download_folder='downloads', year_min=1980, simulation_round=o.simulation_round)


    # Download the files
    def get_file(path, country=None, bbox=None, monthly=not o.daily):

        # special case of a read-only mirror (only unmodified global, daily file are taken from there)
        if o.mirror and not monthly:
            return Path(o.mirror)/path

        target_file = Path(o.download_folder)/path

        if country:
            bbox = boxes[o.country.index(country)]

        if o.download_world:
            tag = "global"

        elif bbox:
            tag = get_region_tag(bbox)

        elif country:
            tag = country

        else:
            tag = "global"


        if "ISIMIP2" in o.simulation_round:
            if monthly:
                timetag = 'month'
            else:
                timetag = 'day'

            return target_file.parent / target_file.name.replace("_global_", f"_{tag}_").replace("_day_", f"_{timetag}_")

        else:
            if monthly:
                timetag = 'monthly'
            else:
                timetag = 'daily'

            return (Path(str(target_file.parent).replace("/global/", f"/{tag}/").replace("/daily/", f"/{timetag}/"))
                / target_file.name.replace("_global_", f"_{tag}_").replace("_daily_", f"_{timetag}_"))


    def download(path, queue=False, country=None, bbox=None, monthly=not o.daily, remove_daily=o.remove_daily, remove_zip=True):
        client = _init_client()
        target_file = get_file(path, country, bbox, monthly=monthly)
        if target_file.exists(): return target_file


        # Monthly files required?
        if monthly:
            # download daily files
            res = download(path, queue, country, bbox, monthly=False)
            if queue and type(res) == dict: return res
            # convert to monthly values
            target_file_daily = res
            os.makedirs(target_file.parent, exist_ok=True)
            cdo(f'monavg {target_file_daily} {target_file}')
            if remove_daily:
                print("rm", target_file_daily)
                os.remove(target_file_daily)
                # remove any empty parent directory
                # needs list of bug (in this Python 3.8.8) https://github.com/python/cpython/issues/79679
                for folder in list(target_file_daily.relative_to(o.download_folder).parents)[:-1]:
                    folder = Path(o.download_folder)/folder
                    if any(folder.glob('*')):
                        break
                    else:
                        print("rm -r", folder)
                        os.rmdir(folder)

            return target_file

        file_url = 'https://files.isimip.org/'+path

        # Convert country to bbox, so we can call cutout (faster processing time than mask)
        if country:
            bbox = boxes[o.country.index(country)]

        kwargs = {} if queue else {'poll': 10}

        if bbox:
            l, b, r, t = bbox
            # job = client.mask(path, bbox=[b, t, l, r], **kwargs) # same download speed, but slower processing (de-compression ?)
            job = client.cutout(path, bbox=[b, t, l, r], **kwargs) # should be faster than mask
            file_url = job['file_url']

        if queue and (bbox or country) and job['status'] != 'finished':
            print(f"Job submitted: {job['id']} ({job['status']})")
            return job

        # Here we can just download the file
        print('download', file_url, 'to', target_file.parent)
        client.download(file_url, path=target_file.parent, validate=False, extract=True)
        assert target_file.exists(), f'something fishy happened: {target_file} does not exist'
        if remove_zip:
            zipfile = target_file.parent/Path(file_url).name
            if zipfile.name.endswith('.zip'):
                print("rm", zipfile)
                os.remove(zipfile)
            readme = target_file.parent/"README.txt"
            if readme.exists():
                print("rm", readme)
                os.remove(readme)
        return target_file


    for r in results:
        for f in r['files']:
            download(f['path'])


    # Now process the files
    print(f"Download and processing to {'daily' if o.daily else 'monthly'} data is done.")
    print("- climate variables:", ' '.join(o.variable))
    print("- climate_forcing:", ' '.join(sorted(set(r['specifiers']['climate_forcing'] for r in results))))
    print("- experiments:", ' '.join(sorted(set(r['specifiers']['climate_scenario'] for r in results))))
    print(f"(total of {len([f for r in results for f in r['files']])} files)")


if __name__ == "__main__":
    main()