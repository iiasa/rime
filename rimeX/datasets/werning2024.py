import os
import tempfile
from pathlib import Path
import subprocess as sp

from rimeX.config import logger
from rimeX.datasets.manager import get_datapath, require_dataset, get_downloadpath

NAME = __name__.split(".")[-1]
DATA = get_datapath(NAME)
_DOWNLOAD = get_downloadpath(NAME)

def download():
    logger.info(f"Download to {_DOWNLOAD}")
    _DOWNLOAD.mkdir(exist_ok=True, parents=True)
    sp.check_call(f"wget -nc -i {DATA}/datasets.txt", shell=True, cwd=_DOWNLOAD)

    logger.info(f"Extract to {DATA}")
    # extract
    for file in _DOWNLOAD.glob("*?download=*"):
        if ".zip" in str(file):
            sp.check_call(f"unzip -n {file.resolve()}", shell=True, cwd=DATA)
