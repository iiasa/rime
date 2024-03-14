import argparse
from pathlib import Path
import yaml

config_parser = argparse.ArgumentParser(add_help=False)
g = config_parser.add_argument_group("config")
g.add_argument("--config-file", default=Path(__file__).parent/"config.yml")

o, _ = config_parser.parse_known_args()

def set_config(file_path):
    global config
    config = yaml.safe_load(open(file_path))

set_config(o.config_file)
    