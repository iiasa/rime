import os
import re
import contextlib, io
from itertools import product, groupby, chain
from pathlib import Path
import argparse
from typing import Optional, List, Union

from rimeX.tools import cdo, check_call
from rimeX.config import CONFIG, config_parser
from rimeX.logs import log_parser, setup_logger, logger


client = None

def _init_client():
    global client
    if client is None:
        from isimip_client.client import ISIMIPClient
        client = ISIMIPClient()
    return client

def _get_simulation_rounds(simulation_round: Optional[Union[str, List[str]]] = None):
    if simulation_round is None:
        simulation_round = CONFIG["isimip.simulation_round"]
    if type(simulation_round) is str:
        simulation_round = [simulation_round]
    return simulation_round

def get_models(simulation_round=None):
    simulation_round = _get_simulation_rounds(simulation_round)
    return list(chain(*(CONFIG["isimip"][simulation_round_]["models"] for simulation_round_ in simulation_round)))

def get_experiments(simulation_round=None):
    simulation_round = _get_simulation_rounds(simulation_round)
    return list(chain(*(CONFIG["isimip"][simulation_round_]["experiments"] for simulation_round_ in simulation_round)))

def get_variables(simulation_round=None):
    return CONFIG["isimip.variables"]

def iterate_model_experiment(simulation_round=None):
    yield from product(get_models(simulation_round), get_experiments(simulation_round))

def _preparse_isimip_protocol():
    """First parse isimip protocol so that defaults models and experiments are correct
    """
    isimip_preparser = _get_isimip_parser(choices=False)
    o, _ = isimip_preparser.parse_known_args()
    if o.simulation_round is not None:
        CONFIG["isimip.simulation_round"] = o.simulation_round

def _get_isimip_parser(choices=True):
    """Determine if --simulation-round is provided in any script that relies on isimip,
    so that get_models() and get_experiments() have the proper choices and default.
    The isimip_parser can be used in any
    """
    isimip_parser = argparse.ArgumentParser(add_help=False)
    group = isimip_parser.add_argument_group('ISIMIP')
    group.add_argument("--experiment", nargs='+', default=get_experiments(), choices=get_experiments() if choices else None)
    _lower_to_upper = {m.lower():m for m in get_models()}
    group.add_argument("--model", nargs='+', default=get_models(), choices=get_models() if choices else None, type=lambda s: _lower_to_upper.get(s, s))
    group.add_argument("--impact-model", nargs='+', default=[None])
    group.add_argument("--simulation-round", nargs="+", default=CONFIG["isimip.simulation_round"], help="default %(default)s") # already set in _prepare_isimip_protocol, but here for help

    return isimip_parser

_preparse_isimip_protocol()
isimip_parser = _get_isimip_parser()

_YEARS_ISIMIP3_RE = re.compile(r'(\d{4})_(\d{4}).nc')
_YEARS_ISIMIP2_RE = re.compile(r'(\d{4})\d{4}-(\d{4})\d{4}.nc')
_YEARS_ISIMIP_RE = re.compile(r'((\d{4})_(\d{4})|(\d{4})\d{4}-(\d{4})\d{4}).nc')

def parse_years(path, simulation_round: Optional[str] = None):
    # if simulation_round is None: simulation_round = CONFIG['isimip.simulation_round']
    simulation_round = None  # some files are different e.g... 'tasmax_gswp3-ewembi_1901_1910.nc4'
    if simulation_round is None:
        groups = _YEARS_ISIMIP_RE.search(Path(path).name).groups()
        if groups[1] is not None:
            y1s, y2s = groups[1], groups[2]
        else:
            y1s, y2s = groups[3], groups[4]
    else:
        _YEARS_RE = _YEARS_ISIMIP2_RE if simulation_round is None or "ISIMIP2" in simulation_round else _YEARS_ISIMIP3_RE
        y1s, y2s = _YEARS_RE.search(Path(path).name).groups()
    y1, y2 = int(y1s), int(y2s)
    return y1, y2

def get_region_tag(bbox):
    l, b, r, t = bbox
    return f"lat{b}to{t}lon{l}to{r}"

# Perhaps useful later on
# def scan_files(variable="*", experiment="*", model="*", download_folder='downloads', year_min=None,
#                simulation_round="*", ensemble="*", bias_adjustment="w5e5", frequency="daily", domain="global", year1="*", year2="*"):
#     # glob_pattern=f"{download_folder}/{simulation_round.upper()}/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp585/GFDL-ESM4/gfdl-esm4_r1i1p1f1_w5e5_ssp585_pr_global_daily_2051_2060.nc"
#     glob_pattern=f"{download_folder}/{simulation_round.upper()}/*/*/*/*/*/{frequency}/{experiment}/{model.upper()}/{model}_{ensemble}_{bias_adjustment}_{experiment}_{variable}_{domain}_{frequency}_{year1}_{year2}.nc"
#     files = sorted(glob.glob(glob_pattern))

def _update_results(results, year_min=None, local_folder=None):

    for r in results:
        for f in r['files']:
            f['time_slice'] = parse_years(f['path'], r['specifiers']['simulation_round'])
            f['local_path'] = get_filepath(r, f['path'], folder=local_folder)
        r['files'] = [f for f in r['files'] if year_min is None or f["time_slice"][1] > year_min]

EXCLUDE_SCENARIOS = ["obsclim", "hist-nat", "picontrol", "counterclim"]


def request_dataset(name, specifiers=None, id=None, climate_forcing=None, climate_scenario=None,
                         year_min=None, simulation_round=None, page_size=1000,
                         exclude_scenarios=EXCLUDE_SCENARIOS, **meta):
    """called isimip_client.client.datasets with config.toml 's indicator metadata
    """
    if id is not None:
        query = {"id": id}
    elif specifiers:
        query = {k: meta[k] for k in specifiers}
    else:
        query = {'climate_variable': name}

    if climate_forcing:
        query['climate_forcing'] = climate_forcing
    else:
        # keep some control in the config file
        query['climate_forcing'] = [m.lower() for m in get_models(simulation_round)]
        # pass

    if climate_scenario:
        query['climate_scenario'] = climate_scenario
    else:
        # otherwise obsclim comes in
        query['climate_scenario'] = get_experiments(simulation_round)
        # pass

    if simulation_round:
        query['simulation_round'] = simulation_round

    if client is None:
        _init_client()

    response = client.datasets(**query, page_size=page_size)
    results = response['results']
    count = response['count']

    if len(results) != count:
        logger.warning(f"Not all results could be fetched. Total count: {count}, fetched: {len(results)}, query: {query}. Modify the page_size limit.")

    # filters out obsclim
    results = [r for r in results if r["specifiers"]["climate_scenario"] not in exclude_scenarios]

    _update_results(results, year_min)

    return results

def aggregate_time(time_aggregation, input_file, output_file, frequency="monthly", dry_run=False):
    """Time aggregation. At the moment, this is designed to
    take a 10-year netcdf file with daily values, and output a monthly values
    instead, with mean, max, min or cumulative values.
    """
    if frequency != "monthly":
        raise NotImplementedError("Only monthly frequency is supported for now")

    if time_aggregation in ("mean", "avg", "", None):
        cdo(f'monavg {input_file} {output_file}', dry_run=dry_run)
    elif time_aggregation in ("max", "min", "sum"):
        cdo(f'mon{time_aggregation} {input_file} {output_file}', dry_run=dry_run)
    else:
        raise ValueError(f"Unknown time aggregation {time_aggregation}")


def download(path, folder=None, queue=False, overwrite=False, remove_zip=True, dry_run=False):

    client = _init_client()

    if folder is None:
        folder = CONFIG["isimip.download_folder"]

    target_file = Path(folder)/path

    if not overwrite and Path(target_file).exists():
        return target_file

    file_url = 'https://files.isimip.org/'+path

    kwargs = {} if queue else {'poll': 10}

    # Here we can just download the file
    print('download', file_url, 'to', target_file.parent)

    if dry_run:
        return target_file

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

def get_filepath(db_result, path, frequency=None, time_aggregation=None, region=None, folder=None):
    """That's for basic ISIMIP files or for indicators that only undergone space or time aggregation
    """
    meta = db_result["specifiers"]

    if frequency is not None:
        orig_freq = meta["time_step"]
        if meta['simulation_round'].startswith("ISIMIP2"):
            rename_freqs = {"daily": "day", "monthly": "month", "annual": "year"}
            # "day" in file name for ISIMIP2
            orig_freq = rename_freqs.get(orig_freq, orig_freq)
            frequency = rename_freqs.get(frequency, frequency)

        if time_aggregation is not None:
            frequency = frequency+"-"+time_aggregation
        path = path.replace(f"_{orig_freq}_", f"_{frequency}_")
        path = path.replace(f"/{orig_freq}/", f"/{frequency}/")

    if region is not None:
        orig_region = meta["region"]
        path = path.replace(f"_{orig_region}_", f"_{region}_")
        path = path.replace(f"/{orig_region}/", f"/{region}/")

    return Path(folder or CONFIG["isimip.download_folder"])/path

def _matches(key1, key2):
    if key1 in (None, "*") or key2 in (None, "*"):
        return True
    if type(key2) is list:
        return any(_matches(key1, k) for k in key2)
    return key1.lower() == key2.lower()


# def get_indicator_path(name, model, experiment, ensemble, frequency, simulation_round=None, bias_correction=None, folder=None):
#     if folder is None:
#         folder = CONFIG["indicators.folder"]
#     ensembletag = f"_{ensemble}" if ensemble else ""
#     biascorrectiontag = f"_{bias_correction}" if bias_correction else ""
#     isimiptag = f"_{simulation_round}" if simulation_round else ""
#     return os.path.join(folder, name, experiment, model, f"{model.lower()}{ensembletag}{biascorrectiontag}{isimiptag}_{experiment}_{name}_{frequency}.nc")


class ISIMIPDataBase:
    def __init__(self, db=[], download_folder=None):
        """
        db: a list of results as returned by _request_isimip_meta
        """
        self.db = db
        self.download_folder = download_folder or CONFIG["isimip.download_folder"]

    def filter(self, **kwargs):
        return [r for r in self.db if all(_matches(r['specifiers'][k], v) for k, v in kwargs.items() if v is not None)]

    def __iter__(self):
        for result in self.db:
            yield result


def _are_consecutive_time_slices(time_slices):
    return all(t2[0] == t1[1]+1 for t1, t2 in zip(time_slices[:-1], time_slices[1:]))


class Indicator:

    def __init__(self, name, frequency="monthly", folder=None,
                 spatial_aggregation=None, depends_on=None, expr=None, time_aggregation=None, isimip_meta=None,
                 shell=None, custom=None,
                 db=None, isimip_folder=None, comment=None, transform=None, year_min=None, units="", title="", projection_baseline=None,
                 depends_on_climatology=False, climatology_quantile=False, historical=True,
                 **kwargs):

        self.name = name
        self.frequency = frequency
        self.expr = expr
        self.shell = shell
        self.custom = custom
        self.folder = Path(folder or CONFIG["indicators.folder"])
        self.isimip_meta = isimip_meta or CONFIG.get(f"indicator.{name}", {}).get("isimip", {})
        for k in kwargs:
            if "." in k:
                parts = k.split(".")
                if parts[0] == "isimip_meta":
                    self.isimip_meta[".".join(parts[1:])] = kwargs[k]
                else:
                    raise ValueError(f"Unknown argument {k}")
            else:
                raise ValueError(f"Unknown argument {k}")

        self.isimip_meta.setdefault("year_min", year_min or CONFIG["isimip.historical_year_min"])

        assert depends_on is None or isinstance(depends_on, list), "depends_on must be a list"
        if isinstance(depends_on, list):
            assert all(type(x) is str for x in depends_on), "depends_on must be a list of strings"
        self.depends_on = depends_on
        self.depends_on_climatology = depends_on_climatology
        self.climatology_quantile = climatology_quantile
        self.spatial_aggregation = spatial_aggregation or CONFIG["preprocessing.regional.weights"]
        self.time_aggregation = time_aggregation
        if isinstance(db, list):
            db = ISIMIPDataBase(db)
        self._db = db
        self._isimip_folder = isimip_folder
        self.comment = comment
        self.transform = transform  # this refers to the baseline period and is only accounted for later on in the emulator (misnamed...)
        self.units = units
        self.historical = historical
        self.title = title
        self.projection_baseline = projection_baseline or CONFIG["preprocessing.projection_baseline"]

    @property
    def ncvar(self):
        return self.isimip_meta.get("variable", self.name)

    @classmethod
    def from_config(cls, name, **kw):
        cfg = CONFIG.get(f"indicator.{name}", {})
        copy = cfg.pop("_copy", None)
        if copy:
            cfg = {**CONFIG.get(f"indicator.{copy}", {}), **cfg}
        if name not in CONFIG["indicator"] and name not in CONFIG["isimip.variables"]:
            raise ValueError(f"Unknown indicator {name}")
        return cls(name, **{**cfg, **kw})

    @property
    def db(self):
        if self._db is None:
            if self.depends_on:
                # self._db = ISIMIPDataBase(sum((_request_isimip_meta(v, **self.isimip_meta) for v in self.depends_on), []), self._isimip_folder)
                dbs = sum((request_dataset(v, **self.isimip_meta) for v in self.depends_on), [])
                # make sure every (experiment, simulation) combination contains data for all dependencies
                key = lambda r: tuple(r['specifiers'][k] for k in self.simulation_keys)
                results = []
                for k, group in groupby(sorted(dbs, key=key), key=key):
                    group = list(group)
                    if len(group) > len(self.depends_on):
                        raise ValueError(f"Something fishy happened. More results found for {self.name} {k} than depends_on {self.depends_on}")
                    if len(group) < len(self.depends_on):
                        logger.debug(f"{[r['specifiers'] for r in group]}")
                        logger.debug(f"{self.name} {k} : at least one dependency missing. Drop.")
                        continue
                    results.extend(group)
                self._db = ISIMIPDataBase(results, self._isimip_folder)
            else:
                self._db = ISIMIPDataBase(request_dataset(self.name, **self.isimip_meta), self._isimip_folder)
        return self._db

    @property
    def simulation_keys(self):
        return ['climate_scenario', 'climate_forcing'] + self.isimip_meta.get("ensemble_specifiers", [])

    @property
    def simulations_values(self):
        # That can be non-unique in case the indicator depends on multiple variables
        return sorted(set(tuple(r['specifiers'][k] for k in self.simulation_keys) for r in self.db))

    @property
    def simulations(self):
        return [dict(zip(self.simulation_keys, values)) for values in self.simulations_values]

    def _get_dataset_meta(self, climate_scenario, climate_forcing, **ensemble_specifiers):
        request_dict = {"climate_forcing": climate_forcing, "climate_scenario": climate_scenario, **ensemble_specifiers}
        for k in self.simulation_keys:
            if k not in request_dict:
                raise ValueError(f"Missing key required {self.name}: {k}. Got {request_dict}")

        results = self.db.filter(**request_dict)

        if len(results) == 0:
            raise ValueError(f"No ISIMIP dataset found for {self.name} {request_dict}")

        checks = [r['specifiers'] for r in results]
        check_str = "\n".join(f"{k}, {[c[k] for c in checks]}" for k in set(checks[0].keys()))

        if self.depends_on:
            assert len(results) == len(self.depends_on), f"{self.name}: Expected {len(self.depends_on)} results, got {len(results)} : \n\n{check_str}"
        else:
            assert len(results) == 1, f"{self.name}: Expected 1 result, got {len(results)} :\n\n{check_str}"

        # Check the time slices are consistent
        time_slices = [f['time_slice'] for f in results[0]['files']]
        if len(results) > 1:
            for r in results[1:]:
                time_slices_ = [f['time_slice'] for f in r['files']]
                if time_slices != time_slices_:
                    print(">>>>")
                    print(results[0]['specifiers'], time_slices)
                    print("====")
                    print(r['specifiers'], time_slices_)
                    print("<<<<")
                    raise RuntimeError("Time slices are not the same for all precursors")

        return results


    def get_path(self, climate_scenario, climate_forcing, region=None, regional=False, regional_weight="latWeight", time_slice=None, **ensemble_specifiers):
        """returns the local file path for this indicator
        """
        result = self._get_dataset_meta(climate_scenario, climate_forcing, **ensemble_specifiers)[0]
        meta = result['specifiers']
        if time_slice is None:
            time_slices = [f['time_slice'] for f in result['files']]
            time_slice = (min(t[0] for t in time_slices), max(t[1] for t in time_slices))
        timeslice_tag = f"_{time_slice[0]}_{time_slice[1]}"
        other_tags = "_".join(meta[k] for k in self.simulation_keys if k not in ["climate_scenario", "climate_forcing"])
        ensemble_tag = f"_{other_tags}" if other_tags else ""

        # one file for each region including admin boundaries
        if region:
            assert regional_weight is not None
            regionfolders = [regional_weight, region]
            region_tag = f"_{region}_{regional_weight}".lower()
            ext = ".csv"

        # one file for all regions
        elif regional:
            region_tag = f"_regional_{regional_weight}".lower()
            regionfolders = []
            ext = ".csv"

        # lon-lat file
        else:
            region_tag = f""
            regionfolders = []
            ext = ".nc"


        # special case where the indicator is exactly the same as the ISIMIP variable
        if not regional and self.frequency == meta["time_step"] and self.depends_on is None and len(time_slices) == 1 and region is None and ext == ".nc":
            with contextlib.redirect_stdout(io.StringIO()):
                return download(result['files'][0]['path'], folder=self.db.download_folder, dry_run=True)

        # otherwise create a new path separate from the ISIMIP database
        basename = f"{climate_forcing.lower()}{ensemble_tag}_{climate_scenario}_{self.name}{region_tag}_{self.frequency}{timeslice_tag}"+ext
        return Path(self.folder, self.name, climate_scenario, climate_forcing.lower(), *regionfolders, basename)

    def download_climatology(self, climate_forcing, dry_run=False, **ensemble_specifiers):
        """compute the climatology of the base variables the indicator depends on
        """
        for name in self.depends_on:
            indicator = Indicator.from_config(name, frequency="daily") # don't do monthly aggregation
            filepath_base = indicator.get_path("historical", climate_forcing, **ensemble_specifiers)
            # below we use the "lazy" download behaviour of self.download, that checks whether a file exists
            if self.climatology_quantile:
                cattedinput = "-cat [ {input} ]"
                filepath_min = Path(str(filepath_base) + f".timmin")
                cat = f"cdo timmin {cattedinput} {filepath_min}"
                indicator.download("historical", climate_forcing, dry_run=dry_run, cat=cat, output_file=filepath_min, **ensemble_specifiers)
                filepath_max = Path(str(filepath_base) + f".timmax")
                cat = f"cdo timmax {cattedinput} {filepath_max}"
                indicator.download("historical", climate_forcing, dry_run=dry_run, cat=cat, output_file=filepath_max, **ensemble_specifiers)
                filepath = Path(str(filepath_base) + f".p{self.climatology_quantile*100}")
                cat = f"cdo timpctl,{self.climatology_quantile*100} {cattedinput} {filepath_min} {filepath_max} {filepath}"
                # cat = f"cdo timpctl,{self.climatology_quantile*100} {cattedinput} -timmin {cattedinput} -timmax {cattedinput} {{output}}"
                indicator.download("historical", climate_forcing, dry_run=dry_run, cat=cat, output_file=filepath, **ensemble_specifiers)
            else:
                filepath = Path(str(filepath_base) + ".climatology")
                cat = f"cdo ymonmean -cat {{input}} {{output}}"
                indicator.download("historical", climate_forcing, dry_run=dry_run, cat=cat, output_file=filepath, **ensemble_specifiers)
            yield filepath

    def download(self, climate_scenario, climate_forcing, time_slice=None, overwrite=False, remove_daily=False, remove_daily_expr=True,
                 cat=None, output_file=None,
                 dry_run=False, **ensemble_specifiers):
        """Download a set of files from the ISIMIP database and returns an iterator on the local file paths (normally over time slices)
        """
        if self.depends_on_climatology:
            clim_files = list(self.download_climatology(climate_forcing, dry_run=dry_run, **ensemble_specifiers))

        if output_file is None:
            output_file = self.get_path(climate_scenario, climate_forcing, **ensemble_specifiers)
        else:
            output_file = Path(output_file)

        if not overwrite and output_file.exists():
            return output_file

        results = self._get_dataset_meta(climate_scenario, climate_forcing, **ensemble_specifiers)
        meta_ = results[0]['specifiers']

        temporary_files = set()

        def _mark_for_cleanup(files):
            for f in files:
                temporary_files.add(str(f))

        def _cleanup(excludes=[]):
            # clean up the temporary files if any, to save disk space
            while len(temporary_files) > 0:
                tmp = temporary_files.pop()
                if Path(tmp).exists() and str(tmp) not in map(str, excludes):
                    check_call(f"rm '{tmp}'", dry_run=dry_run)


        time_slices = [f['time_slice'] for f in results[0]['files']]
        assert _are_consecutive_time_slices(time_slices), f"Time slices are not consecutive: {time_slices}"

        time_slice_files = []
        previous_input_files = None
        previous_output_file = None

        for t, time_slice in enumerate(time_slices):

            # temporary monthly (or else time-aggregated) file
            if not self.depends_on:
                # for the classical ISI-MIP variables, use the same file pattern as downloaded file for temporary files
                # this is for legacy reasons, to re-use the files that are already downloaded and created this way
                # we may rename that later on and only keep the simpler form below
                files = [f for f in results[0]['files'] if f['time_slice'] == time_slice]
                assert len(files) == 1, f"Expected 1 file, got {len(files)}"
                time_slice_file = get_filepath(results[0], files[0]['path'], frequency=self.frequency, time_aggregation=self.time_aggregation, folder=self.db.download_folder)

            else:
                time_slice_file = self.get_path(climate_scenario, climate_forcing, time_slice=time_slice, **ensemble_specifiers)

            if not overwrite and time_slice_file.exists():
                time_slice_files.append(time_slice_file)
                continue

            if not dry_run:
                time_slice_file.parent.mkdir(parents=True, exist_ok=True)

            # download input (generally daily) files
            # as many results as input variables (1 if depends_on is None)
            input_daily_files = []

            for result in results:
                files = [f for f in result['files'] if f['time_slice'] == time_slice]
                assert len(files) == 1, f"Expected 1 file, got {len(files)}"
                f = files[0]
                local_file = download(f['path'], folder=self.db.download_folder, dry_run=dry_run, overwrite=overwrite)
                input_daily_files.append(local_file)

            # for `expr` and for convenience as a `shell` command placeholder we define the input file
            # via -merge in case the indicator depends on multiple variables
            if len(input_daily_files) > 1:
                input_file = f"-merge {' '.join(map(str, input_daily_files))}"

            else:
                input_file = input_daily_files[0]

            # Any shell command (this can also be cdo)
            if self.shell:
                # a command with placeholders:
                # {inputs} : a list of input files
                # {input} : joined inputs with " " separator
                # {output} : the output file
                # {previous_inputs} : {inputs} from the previous time slice
                # {previous_input} : joined {previous_inputs} with " " separator
                # {previous_output} : the first element of {previous_outputs}
                # {name}
                cmd = self.shell.format(
                                        inputs=input_daily_files,
                                        input=" ".join(input_daily_files),
                                        output=time_slice_file,
                                        previous_inputs=previous_input_files,
                                        previous_input=" ".join(previous_input_files) if previous_input_files else "",
                                        previous_output=previous_output_file,
                                        name=self.name)

                check_call(cmd, dry_run=dry_run)
                if not dry_run:
                    assert Path(time_slice_file).exists(), f"Shell command {cmd} did not create {time_slice_file}"

            # custom function
            elif self.custom:
                import importlib
                module, function = self.custom.split(":")
                custom_module = importlib.import_module(module)
                func = getattr(custom_module, function)
                kwargs = dict(
                    previous_input_files=previous_input_files,
                    previous_output_file=previous_output_file,
                    dry_run=dry_run,
                    )
                if self.depends_on_climatology:
                    func(input_daily_files, clim_files, time_slice_file, **kwargs)
                else:
                    func(input_daily_files, time_slice_file, **kwargs)

                if not dry_run:
                    assert Path(time_slice_file).exists(), f"Custom function {self.custom} did not create {time_slice_file}"

            else:

                # Calculate any expression into another daily file
                if self.expr:
                    # this file only exists at some point if the indicator is computed from expr, otherwise it is renamed later
                    time_slice_file_daily = Path(str(time_slice_file) + ".daily")

                    if not time_slice_file_daily.exists():

                        if "=" not in self.expr:
                            expr = f"{self.name}={self.expr}"
                        else:
                            expr = self.expr

                        cdo(f"expr,'{expr}' {input_file} {time_slice_file_daily}", dry_run=dry_run)

                        if remove_daily or remove_daily_expr:
                            _mark_for_cleanup([time_slice_file_daily])

                # standard variables e.g. tas, tasmax that come straight from ISIMIP
                else:
                    assert len(input_daily_files) == 1, "Expected 1 input file when no expr exists"
                    time_slice_file_daily = input_daily_files[0]


                # time aggregation
                if self.frequency != meta_["time_step"]:
                    aggregate_time(self.time_aggregation, time_slice_file_daily, time_slice_file, frequency=self.frequency, dry_run=dry_run)

                else:
                    # e.g. crop yield has an annual time_step
                    assert time_slice_file == time_slice_file_daily, (time_slice_file, time_slice_file_daily)

            _cleanup(excludes=[time_slice_file])

            time_slice_files.append(time_slice_file)

            previous_input_files = input_daily_files
            previous_output_file = time_slice_file

            if remove_daily:
                # this will be removed after the next pass because some functions need the previous_input_files
                _mark_for_cleanup(input_daily_files)


        # custom concatenation of time slice files (e.g. for quantiles)
        if cat is not None:
            if not dry_run:
                output_file.parent.mkdir(parents=True, exist_ok=True)
            input_files = " ".join(map(str, time_slice_files))
            catcmd = cat.format(input=input_files, output=output_file) # {input} and {output} are placeholders
            check_call(catcmd, dry_run=dry_run)
            if self.frequency != "daily" or remove_daily:
                _mark_for_cleanup(time_slice_files)

        elif len(time_slice_files) == 1:
            if time_slice_files[0] != output_file:
                check_call(f"mv '{time_slice_files[0]}' '{output_file}'", dry_run=dry_run)

        else:
            if not dry_run:
                output_file.parent.mkdir(parents=True, exist_ok=True)
            cdo(f"cat {' '.join(map(str, time_slice_files))} {output_file}", dry_run=dry_run)
            if self.frequency != "daily" or remove_daily:
                _mark_for_cleanup(time_slice_files)

        _cleanup(excludes=[output_file])

        return output_file


    def download_all(self, **kwargs):
        for simu in self.simulations:
            yield self.download(**simu, **kwargs)

    def get_all_paths(self, filter=None, **kwargs):
        for simu in self.simulations:
            if filter is not None:
                if not filter(simu):
                    continue
            yield self.get_path(**simu, **kwargs)


def main():
    all_variables = list(CONFIG["isimip.variables"]) + sorted(set(v.split(".")[0] for v in CONFIG["indicator"]))

    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    # parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"], help='the original ISIMIP variables')
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=all_variables)
    parser.add_argument("--daily", action='store_true', dest='daily', help=argparse.SUPPRESS)
    # parser.add_argument("--monthly-statistic", default=["mean"], nargs="*",
    #                    help="""function(s) to process daily into monthly variables. Default is ["mean"].
    #                    It includes predfined functions: ["mean", "std", "max", "min", "sum"]. TODO: user-defined functions via mdoule import""")
    # parser.add_argument("--keep-daily", action='store_false', dest='remove_daily', help=argparse.SUPPRESS)
    parser.add_argument("--remove-daily", action='store_true')
    parser.add_argument("--mirror", help=argparse.SUPPRESS)  # in case we have direct access to PIK cluster, say
    parser.add_argument("--download-folder", default=CONFIG["isimip.download_folder"], help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action='store_true')

    o = parser.parse_args()
    setup_logger(o)

    CONFIG["isimip.download_folder"] = o.download_folder

    if not (o.indicator):
        parser.error("Please provide either --variable or --indicator")
        parser.print_help()
        parser.exit(1)

    print("model", o.model)
    print("experiment", o.experiment)

    if o.indicator:
        for name in o.indicator:
            indicator = Indicator.from_config(name)
            for simu in indicator.simulations:
                for model, experiment in iterate_model_experiment():
                    if _matches(simu["climate_scenario"], experiment) and _matches(simu["climate_forcing"], model):
                        print(f"Downloading {name} for {simu}")
                        indicator.download(**simu, overwrite=o.overwrite, remove_daily=o.remove_daily)

if __name__ == "__main__":
    main()