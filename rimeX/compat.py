import fnmatch
import numpy as np
import pandas as pd
from rimeX.logs import logger

def read_table(file, backend=None, index=["model", "scenario"], **kwargs):
    logger.info(f"Read {file}")
    if backend == "feather" or str(file).endswith((".ftr", ".feather")):
        df = pd.read_feather(file, **kwargs)
    elif backend == "excel" or str(file).endswith((".xls", ".xlsx")):
        with pd.ExcelFile(file) as xls:
            df = pd.read_excel(xls, **kwargs)

            # Also consider the "meta" columns, similarly to pyam
            if "meta" in xls.sheet_names:
                idx_cols = FastIamDataFrame(df, index=index)._index
                meta = pd.read_excel(xls, "meta", **kwargs).set_index(idx_cols)
                df2 = df.set_index(idx_cols)
                # put meta names first to keep the years at the end
                df = df2.join(meta.loc[df2.index])[list(meta.columns) + list(df2.columns)].reset_index()

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


class FastIamDataFrame:
    def __init__(self, df, index=["model", "scenario"]):
        logger.debug(f"FastIamDataFrame input df columns {df.columns}")
        self.df = df
        self._index = [self._get_col_name(c) for c in index]

    def _get_col_name(self, name):
        lowercolumns = [c.lower() for c in self.df.columns]
        return self.df.columns[lowercolumns.index(name.lower())] # case-proof

    def _get_col(self, name):
        return self.df[self._get_col_name(name)]

    @property
    def variable(self):
        return sorted(self._get_col("variable").unique())

    @property
    def model(self):
        return sorted(self._get_col("model").unique())

    @property
    def scenario(self):
        return sorted(self._get_col("scenario").unique())

    @property
    def region(self):
        return sorted(self._get_col("region").unique())

    @property
    def index(self):
        return self.df.set_index(self._index).index.unique()

    def _first_year_index(self):
        columns = self.df.columns
        numerical_types = np.array([_isnumerical(c) for c in columns])
        i_last_meta = np.where(~numerical_types)[0][-1]
        i_first_year = i_last_meta + 1
        return i_first_year

    @property
    def year(self):
        return [_to_numerical(y) for y in self.df.columns[self._first_year_index():]]

    @classmethod
    def load(cls, file, **kwargs):
        df = read_table(file)
        return cls(df)

    def __len__(self):
        return len(self.df) * len(self.year)

    def rename(self, inplace=False, **kwargs):
        if inplace:
            self.df.rename(inplace=True, **kwargs)
        else:
            return FastIamDataFrame(self.df.rename(**kwargs), index=self._index)



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

        # we actually went through the loop
        if andcond is not True:
            df = df[andcond] # actually indexing

        res = FastIamDataFrame(df, index=self._index)

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
        return FastIamDataFrame(df, index=self._index)

    def as_pandas(self):
        columns = self.df.columns
        i_first_year = self._first_year_index()
        years = columns[i_first_year:]
        if len(years) == 0:
            # no year present
            return self.df
        meta = columns[:i_first_year]
        df_meta = self.df[meta].rename({name:name.lower() for name in self.df.columns}, axis=1)
        return pd.concat(df_meta.join(pd.DataFrame({"year": _to_numerical(year), "value": self.df[year]})) for year in years).set_index([c.lower() for c in self._index]).reset_index()

    @property
    def dimensions(self):
        return [c for c in self.df.columns if c not in ['value', 'unit'] and not _isnumerical(c)]


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
        info = f"{type(self)}\nIndex:\n"
        c1 = max([len(i) for i in self.dimensions]) + 1
        c2 = n - c1 - 5
        info += "\n".join(
            [
                f" * {i:{c1}}: {print_list(getattr(self.df, i).unique(), c2)}"
                for i in self.index.names
            ]
        )

        # concatenate list of index of _data (not in index.names)
        info += "\nOther columns:\n"
        info += "\n".join(
            [
                f"   {i:{c1}}: {print_list(getattr(self.df, i).unique(), c2)}"
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