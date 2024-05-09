"""This module compute ISIMIP GCMs global mean temperature w.r.t. the projection baseline, to temperature matching.
"""
import argparse
import os
from pathlib import Path
import xarray as xa

from rimeX.preproc.regional_average import get_files
from rimeX.tools import cdo
from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser


def global_mean_file(variable, model, experiment, root=None):
    if root is None: root = CONFIG["isimip.climate_impact_explorer"]
    return Path(root) / f"isimip_global_mean/{variable}/globalmean_{variable}_{model.lower()}_{experiment}.csv"


def main():
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser])
    # parser.add_argument("--variable", nargs="+", default=["tas"], choices=["tas"])
    parser.add_argument("--model", nargs="+", default=CONFIG["isimip.models"], choices=CONFIG["isimip.models"])
    parser.add_argument("--experiment", nargs="+", default=CONFIG["isimip.experiments"], choices=CONFIG["isimip.experiments"])
    o = parser.parse_args()
    setup_logger(o)

    variable = "tas"

    for model in o.model:
        for experiment in o.experiment:

            ofile = global_mean_file(variable, model, experiment)

            if os.path.exists(ofile):
                logger.info(f"{model} | {experiment} :: {ofile} already exists")
                continue

            input_files = get_files(variable, model, experiment)

            if not input_files:
                logger.warning(f"No files found for {model} | {experiment} ...")
                continue

            logger.info(f"Process {model} | {experiment} ...")

            files = []
            for file in input_files:
                ofiletmp = Path(file.replace("global", "globalmean"))
                if not ofiletmp.exists():
                    ofiletmp.parent.mkdir(exist_ok=True, parents=True)
                    cdo(f"-O -fldmean -selname,{variable} {file} {ofiletmp}")
                files.append(ofiletmp)

            with xa.open_mfdataset(files, concat_dim="time", combine="nested", cache=False) as mfd:
                tas = mfd[variable].load()

            logger.info(f"Write to {ofile}")
            ofile.parent.mkdir(exist_ok=True, parents=True)
            s = tas.squeeze().to_pandas()
            s.name = variable
            s.to_csv(ofile)


if __name__ == "__main__":
    main()
