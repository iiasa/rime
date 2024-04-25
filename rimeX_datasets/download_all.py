import os
from pathlib import Path
import subprocess as sp
datadir = Path(__file__).parent
download = datadir.parent / "rimeX_datasets.downloads"
sp.check_call(f"wget -nc -i {datadir}/datasets.txt", shell=True, cwd=download)
for file in download.glob("*?download=*"):
    if ".zip" in str(file):
        sp.check_call(f"unzip -n {file.resolve()}", shell=True, cwd=datadir)
