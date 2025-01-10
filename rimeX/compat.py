import fnmatch
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xa
from rimeX.logs import logger
from rimeX.datasets import get_datapath
from rimeX.config import CONFIG

def read_table(file, backend=None, index=None, **kwargs):
    logger.info(f"Read {file}")
    if backend == "feather" or str(file).endswith((".ftr", ".feather")):
        df = pd.read_feather(file, **kwargs)
    elif backend == "excel" or str(file).endswith((".xls", ".xlsx")):
        with pd.ExcelFile(file) as xls:
            df = pd.read_excel(xls, **kwargs)

            # Also consider the "meta" columns, similarly to pyam
            if "meta" in xls.sheet_names:
                meta = pd.read_excel(xls, "meta", **kwargs)
                # idx_cols = [c for c in FastIamDataFrame(df, index=index)._index_names if c in meta.columns]
                idx_cols = [c for c in df.columns if c in meta.columns]
                logger.info(f"Join meta based on index {idx_cols}")
                meta = meta.set_index(idx_cols)
                df = df.set_index(idx_cols)
                nlen = len(df)
                df = pd.concat([meta.loc[df.index], df], axis=1).reset_index()
                assert len(df) == nlen, 'an issue arose during merging metadata'


    elif backend == "parquet" or str(file).endswith((".parquet")):
        df = pd.read_parquet(file, **kwargs)
    else:
        df = pd.read_csv(file, **kwargs)
    return df


def print_list(x, n):
    """Return a printable string of a list shortened to n characters

    Source: copied from pyam.utils
    """
    # if list is empty, only write count
    if len(x) == 0:
        return "(0)"

    # write number of elements, subtract count added at end from line width
    x = [i if i != "" else "''" for i in map(str, x)]
    count = f" ({len(x)})"
    n -= len(count)

    # if not enough space to write first item, write shortest sensible line
    if len(x[0]) > n - 5:
        return "..." + count

    # if only one item in list
    if len(x) == 1:
        return f"{x[0]} (1)"

    # add first item
    lst = f"{x[0]}, "
    n -= len(lst)

    # if possible, add last item before number of elements
    if len(x[-1]) + 4 > n:
        return lst + "..." + count
    else:
        count = f"{x[-1]}{count}"
        n -= len({x[-1]}) + 3

    # iterate over remaining entries until line is full
    for i in x[1:-1]:
        if len(i) + 6 <= n:
            lst += f"{i}, "
            n -= len(i) + 2
        else:
            lst += "... "
            break

    return lst + count


def _simplify(name):
    if type(name) is not str:
        return str(name)
    name = name.replace("_","").replace("-","").replace(".", "").lower()
    if name in ("warminglevel", "gwl", "gmt"):
        name = "warming_level"
    elif name in ("sspfamily", "ssp"):
        name = "ssp_family"
    elif name in ("experiment"):
        name = "scenario"
    return name


def get_rename_mapping(names):
    simplified = [_simplify(nm) for nm in names]
    if len(set(simplified)) != len(names):
        logger.error(f"input names: {names}")
        logger.error(f"would be renamed to: {simplified}")
        raise ValueError("some column names are duplicate or ambiguous")

    return dict((k,v) for k,v in zip(names, simplified) if k != v)


def _get_ssp_mapping(scenarios):
    """Returns a common mapping for SSP family, in the form of ["ssp1", etc..]
    """
    if _isnumerical(scenarios[0]):
        # return [f"ssp{int(s)}" for s in scenarios]
        return scenarios
    else:
        return [int(s[3]) for s in scenarios]


def _parse_warming_level_and_ssp(scenarios):
    """ Parse warming levels and SSP family from the Werning et al scenarios

    Parameters
    ----------
    scenarios: array-like of type "ssp1_2p5"

    Returns
    -------
    warming_levels: float, array-like (global warming levels)
    scenarios: list of strings ["ssp1", ...]
    """
    warming_levels = np.empty(len(scenarios))
    ssp_family = []
    try:
        for i, value in enumerate(scenarios):
            ssp, gwl = value.split("_")
            warming_levels[i] = float(gwl.replace('p', '.'))
            ssp_family.append(ssp)
    except:
        logger.error(f"Expected scenario such as ssp1_2p0 to derive warming_level. Got: {value}")
        raise

    return warming_levels, ssp_family


def homogenize_table_names(df):
    """Make sure loosely named input table names are understood by the script, e.g. WarmingLevel or gwl or gmt => warming_level,
    and retrieve the warming level from the scenario name if need be.

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    DataFrame
    """
    names = df.columns

    mapping = get_rename_mapping(names)

    df = df.rename(mapping, axis=1)

    names = df.columns

    # ADD 'warming_level' threshold if absent. For now assume scenarios like ssp1_2p0 ==> warming level = 2.0
    # ...also replace scenario with the ssp scenario only
    if "warming_level" not in names:
        assert 'scenario' in names, "Input table must contain `warming_level` or a `scenario` column of the form `ssp1_2p0`"
        df['warming_level'], df['scenario'] = _parse_warming_level_and_ssp(df["scenario"].values)

    # # Also add missing fields that are not actually mandatory but expected in various subfunctions
    # for field in ["variable", "region", "scenario", "model"]:
    #     if field not in df:
    #         df[field] = ""

    return df


class FastIamDataFrame:
    """Inspired by pyam's IamDataFrame but faster

    - accept a pandas DataFrame as input parameter
    - read via FastIamDataFrame.load method
    - keeps wide format in memory
    - convert to long with as_pandas()

    Note it was put together quickly without a deep understanding of pyam's IamDataFrame, so we surely miss a lot of things.
    But convenient to just load the data and apply a `filter` method.
    """
    def __init__(self, df, index=None):
        logger.debug(f"FastIamDataFrame input df columns {df.columns}")
        self.df = df
        if index is not None:
            # self._index_names = [self._get_col_name(c) for c in index]
            self._index_names = index
        else:
            self._index_names = self.dimensions
        self._mapping = get_rename_mapping(df.columns)
        self._inv_mapping = {v:k for k,v in self._mapping.items()}
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = self.df.set_index(self._index_names).index
        return self._index
        # # self.index = full_index.unique()
        # self.index = full_index.unique()
        # # if len(full_index) > len(self.index):
        # #     logger.warning(f"FastIamDataFrame: the index is not unique {len(full_index)} > {len(self.index)}\n{full_index}")

    def rename(self, **kwargs):
        return FastIamDataFrame(self.df.rename(kwargs, axis=1), index=[kwargs.get(c, c) for c in self._index_names])

    def standardize(self):
        if not self._mapping:
            return self
        logger.info(f"FastIamDataFrame: rename columns: {self._mapping}")
        return self.rename(**self._mapping)
        # return FastIamDataFrame(self.df.rename(self._mapping, axis=1), [self._mapping.get(c, c) for c in self._index_names])

    def _get_col_name(self, name):
        return self._inv_mapping.get(name, name)

    def _get_col(self, name):
        return self.df[self._get_col_name(name)]

    def __getattr__(self, name):
        if name in self.df.columns:
            return self.df[name].unique()
        else:
            raise AttributeError(name)

    def _first_year_index(self):
        columns = self.df.columns
        numerical_types = np.array([_isnumerical(c) for c in columns])
        i_last_meta = np.where(~numerical_types)[0][-1]
        i_first_year = i_last_meta + 1
        return i_first_year

    @property
    def year(self):
        return [_to_numerical(y) for y in self.df.columns[self._first_year_index():]]

    def is_wide(self):
        return len(self.year) > 0

    @classmethod
    def load(cls, file, **kwargs):
        df = read_table(file)
        return cls(df)

    def __len__(self):
        return len(self.df)

    def filter(self, **kwargs):

        if not kwargs:
            return self

        df = self.df
        columns = self.df.columns
        lowercolumns = [c.lower() for c in columns]
        andcond = True
        for k_, exprs in kwargs.items():
            if k_.lower() == "year": continue
            k = self._get_col_name(k_)
            values = df[k].unique()
            if type(exprs) is not list:
                exprs = [exprs]
            keep = []
            for expr in exprs:
                keep.extend(fnmatch.filter(values, expr) if type(expr) is str else [v for v in values if v == expr])
            orcond = False
            for value in keep:
                orcond = (df[k].values == value) | orcond
            andcond = orcond & andcond
            if andcond is False:
                raise ValueError(f"No data match the filter: {k_} = {exprs}. Values found: {set(values)}.")

        # we actually went through the loop
        if andcond is not True:
            if andcond is False:
                raise ValueError("No data match the required filter")
            df = df[andcond] # actually indexing

        res = FastIamDataFrame(df, index=self._index_names)

        if "year" in kwargs:
            return res._filter_year(kwargs['year'])
        elif "Year" in kwargs:
            return res._filter_year(kwargs['Year'])

        return res

    def _filter_year(self, year):
        if _isnumerical(year):
            year = [year]
        iyear = self._first_year_index()
        year_cols = self.df.columns[iyear:]
        cols = [c for c in year_cols if str(c) in year or _to_numerical(c) in year]
        all_cols = list(self.df.columns[:iyear]) + cols
        df = self.df[all_cols]
        return FastIamDataFrame(df, index=self._index_names)


    def to_long(self):
        columns = self.df.columns
        i_first_year = self._first_year_index()
        years = columns[i_first_year:]
        if len(years) == 0:
            # no year present
            return self

        meta = columns[:i_first_year]
        df_meta = self.df[meta]
        return FastIamDataFrame(pd.concat(df_meta.join(pd.DataFrame({"year": _to_numerical(year), "value": self.df[year]})) for year in years), index=self.index.names+['year'])


    def as_pandas(self):
        """ Returned a standardized long-form pandas dataframe
        """
        return self.standardize().to_long().df


    @property
    def dimensions(self):
        """Returns an index
        """
        return [c for c in self.df.columns if c not in CONFIG["index.ignore"] and not _isnumerical(c)]


    def info(self, n=80):
        """Print a summary of the object index dimensions and meta indicators

        Parameters
        ----------
        n : int
            The maximum line length

        Source
        ------
        This is adapted from pyam's IamDataFrame's info method
        """

        # concatenate list of index dimensions and levels
        info = f"{type(self)}\nIndex ({len(self.index)}):\n"
        c1 = max([len(i) for i in self.dimensions]) + 1
        c2 = n - c1 - 5
        info += "\n".join(
            [
                f" * {i:{c1}}: {print_list(self.df[i].unique(), c2)}"
                for i in self.index.names
            ]
        )

        if self.is_wide():
            info += "\nYear in columns:\n"
            info += f" * {'year':{c1}}: {print_list(self.year, c2)}"

        # concatenate list of index of _data (not in index.names)
        if len([i for i in self.dimensions if i not in self.index.names]):
            info += "\nOther columns:\n"
            info += "\n".join(
                [
                    f"   {i:{c1}}: {print_list(self.df[i].unique(), c2)}"
                    for i in self.dimensions
                    if i not in self.index.names
                ]
            )

        return info


    def __repr__(self):
        return self.info()


def _to_numerical(year):
    try:
        return int(year)
    except TypeError:
        return float(year)

def _isnumerical(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def concat(fastiamdfs, **kwargs):
    return FastIamDataFrame(pd.concat([elem.df for elem in fastiamdfs], **kwargs), index=fastiamdfs[0]._index)



def _get_custom_filters(groups, mapping=None):
    all_filters = []
    for group in groups:
        kw = {}
        for k, v in group:
            if mapping:
                k = mapping.get(k, k)
            kw.setdefault(k, [])
            kw[k].append(v)
        all_filters.append(kw)
    return all_filters



def load_file(file, and_filters={}, or_filters=[], index=None,
    choices=None, rename=True, pyam=False):
    """Load data files in table form (primarily for command-line use)

    Parameters
    ----------
    file: path to an existing file (read directly)

    and_filter: dictionary of key-word arguments to filter the loaded dataframe
        e.g. model, scenario or else. Both the key-word arguments and the loaded
        dataframe will have their variable names simplified before application
        (lower-case, hypthen, space and underscore removed; some fields renamed e.g. experiment -> scenario)

    or_filter: list of dictionaries of key-word arguments that will be joined together

    index: if provided, identify the fields from input dataset that identify a unique key
        otherwise, the keys in "choices" are tried

    choices: possible keys for an index

    rename: bool
        if True, standardize the variable names

    Returns
    -------
    FastIamDataFrame or IamDataFrame (if pyam=True)

    """

    df_wide = read_table(file)

    remap = get_rename_mapping(df_wide.columns)
    if remap:
        df_wide = df_wide.rename(remap, axis=1)

    if index is not None:
        index = [_simplify(c) for c in index]

    if pyam:
        import pyam.core
        concat_func = pyam.core.concat
        iamdf_cls = pyam.core.IamDataFrame
        iamdf = pyam.core.IamDataFrame(df_wide, index=index)
    else:
        concat_func = concat
        iamdf_cls = FastIamDataFrame
        iamdf = FastIamDataFrame(df_wide, index=index)
        index = iamdf.index.names  # update index

    # logger.debug(f"Before filtering:\n{iamdf}")

    ## The variables below are applies as an outer product (e.g. 3 variables x 2 models x 1 scenario = 6 items)
    filter_kw = {}

    for dim, values in and_filters.items():
        if values is not None:
            filter_kw[remap.get(dim, dim)] = values
        # for v in values:
        #     if v is not None:
        #         filter_kw.setdefault(dim, [])
        #         filter_kw[dim].append(v)

    iamdf_filtered = iamdf.filter(**filter_kw)

    assert len(iamdf_filtered) > 0, f"{filter_kw} resulted in 0-length datarray"

    # logger.debug(f"After AND filters:\n{iamdf_filtered}")

    # additionally, use --gsat-filter to combine groups of arguments in an additive manner
    custom_filters = _get_custom_filters(or_filters)
    if custom_filters:
        dfs = [iamdf_filtered.filter(**{_simplify(k):v for k,v in kw.items()}).as_pandas() for kw in custom_filters]
        # this create duplicates => must be drop afterwardsthis should handle duplicates
        if len(custom_filters) > 1:
            df = pd.concat(dfs).drop_duplicates()
        else:
            df = dfs[0]
        iamdf_filtered = iamdf_cls(df, index=index)

    return iamdf_filtered



def load_files(files, pyam=False, **kwargs):
    """Wrapper around load_file that handles a list (of files use)

    Parameters for command-line
    ----------
    files: list
        each file can be either an existing file on disk, or a glob* pattern within the dataset folder (via get_datapath)
    **kwargs: key-word arguments (to load_file use)

    Returns for command-line
    -------
    FastIamDataFrame or IamDataFrame (if pyam=True)
    """
    if pyam:
        from pyam.core import concat as concat_func
    else:
        concat_func = concat

    scanned_files = []
    for file in files:
        if Path(file).exists():
            scanned_files.append(file)
        else:
            scanned_files.extend(sorted(glob.glob(str(get_datapath(file)))))

    if len(scanned_files) == 0:
        raise FileNotFoundError(f"No file(s) found for {files}")

    return concat_func([load_file(file, pyam=pyam, **kwargs) for file in scanned_files])



def _open_dataset(func, file, **kwargs):
    """Wrapp xarray.open_mfdataset but handles the "years since..." time format.
    """
    try:
        ds = func(file, **kwargs)

    except ValueError:
        ds = func(file, decode_times=False, **kwargs)
        if ds["time"].units.startswith("years since"):
            firstyear = int(ds["time"].units[len("years since "):].split("-")[0])
            ds["time"] = pd.date_range(start=f"{ds['time'].values[0]+firstyear}", periods=ds["time"].size, freq='A')
        else:
            logger.warning(f"Cannot decode time: {file}")
            raise

    return ds


def open_dataset(file, **kwargs):
    return _open_dataset(xa.open_dataset, file, **kwargs)


def open_mfdataset(files, **kwargs):
    return _open_dataset(xa.open_mfdataset, files, **kwargs)
