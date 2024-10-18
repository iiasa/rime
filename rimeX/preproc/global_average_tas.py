"""This module compute ISIMIP GCMs global mean temperature w.r.t. the projection baseline, to temperature matching.
"""
import argparse
import os
from pathlib import Path
import xarray as xa

from rimeX.datasets.download_isimip import get_models, get_experiments
from rimeX.preproc.regional_average import get_files, isimip_parser
from rimeX.tools import cdo
from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser


def global_mean_file(variable, model, experiment, simulation_round=None, root=None):
    if simulation_round is None: simulation_round = CONFIG["isimip.simulation_round"]
    if root is None: root = Path(CONFIG["isimip.climate_impact_explorer"]) / {"ISIMIP2b":"isimip2", "ISIMIP3b":"isimip3"}.get(simulation_round, simulation_round)
    return Path(root) / f"isimip_global_mean/{variable}/globalmean_{variable}_{model.lower()}_{experiment}.csv"


def load_global_average_tas(variable, model, experiment, simulation_round=None, csv=True):

    if csv:
        ofile = global_mean_file(variable, model, experiment, simulation_round)

        if os.path.exists(ofile):
            logger.info(f"{model} | {experiment} :: {ofile} already exists")
            import pandas as pd
            return pd.read_csv(ofile, index_col=0)

    input_files = get_files(variable, model, experiment, simulation_round=simulation_round)

    if not input_files:
        logger.warning(f"No files found for {model} | {experiment} ...")
        return
        # continue

    logger.info(f"Process {model} | {experiment} ...")

    files = []
    for file in input_files:
        ofiletmp = Path(file.replace("global", "globalmean"))
        if not ofiletmp.exists():
            ofiletmp.parent.mkdir(exist_ok=True, parents=True)
            cdo(f"-O -fldmean -selname,{variable} {file} {ofiletmp}")
        files.append(ofiletmp)

    # with xa.open_mfdataset(files, concat_dim="time", combine="nested") as mfd:
    #     tas = mfd[variable].load().squeeze()
    tas = xa.concat([xa.open_dataset(f)[variable].load() for f in files], dim="time").squeeze() # 30% faster than the above

    if csv:
        logger.info(f"Write to {ofile}")
        ofile.parent.mkdir(exist_ok=True, parents=True)
        s = tas.to_pandas()
        s.name = variable
        s.to_csv(ofile)

    return tas

def main():
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, isimip_parser])
    parser.add_argument("--no-csv", dest="csv", action="store_false", help="save/load to csv for speed-up")
    o = parser.parse_args()
    setup_logger(o)

    variable = "tas"

    for model in o.model:
        for experiment in o.experiment:
            load_global_average_tas(variable, model, experiment, o.simulation_round, csv=o.csv)


if __name__ == "__main__":
    main()
