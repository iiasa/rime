import os
import tempfile
from pathlib import Path
import subprocess as sp

from rimeX.config import logger
from rimeX.datasets.manager import register_dataset

require_ar6_wg3 = register_dataset("AR6-WG3-plots/spm-box1-fig1-warming-data.csv", 
    "https://zenodo.org/records/6496232/files/data/raw/spm-box-1-fig-1/spm-box1-fig1-warming-data.csv?download=1", 
    doi="10.5281/zenodo.6420006")

require_ar6_wg3_lhs = register_dataset("AR6-WG3-plots/spm-box1-fig1-warming-data-lhs.csv", 
    "https://zenodo.org/records/6496232/files/data/processed/spm-box-1-fig-1/spm-box1-fig1-warming-data-lhs.csv?download=1", 
    doi="10.5281/zenodo.6420006")


def read_raw_data():
    import pd, pyam
    cols = list(pd.read_csv(require_ar6_wg3(), nrows=1))
    return pyam.IamDataFrame(require_ar6_wg3(), usecols=cols[cols.index('exclude')+1:])


def read_processed_data():
    import pyam
    return pyam.IamDataFrame(require_ar6_wg3_lhs())