import argparse
from pathlib import Path
import yaml

from rimeX.logs import logger

DEFAULT_CONFIG_FILE = Path(__file__).parent/"config_isimip3.yml"
DEFAULT_CONFIG = yaml.safe_load(open(DEFAULT_CONFIG_FILE))

def search_default_config():
    # seach current working directory
    candidates = ['rimeX.yml', 'rime.yml', 'config.yml']
    for candidate in candidates:
        if Path(candidate).exists():
            logger.info(f"Found config file: {candidate}")
            return candidate

    # not found, so return default
    logger.debug(f"No config file found, use defaults.")
    return DEFAULT_CONFIG_FILE

config_parser = argparse.ArgumentParser(add_help=False)
g = config_parser.add_argument_group("config")
g.add_argument("--version", action='store_true')
g.add_argument("--config-file", default=search_default_config())

o, _ = config_parser.parse_known_args()

if o.version:
    from rimeX._version import __version__
    print(__version__)
    config_parser.exit(0)


def set_config(file_path):
    global config
    config = yaml.safe_load(open(file_path))

    # update undefined fields with defaults
    for field in DEFAULT_CONFIG:
        config.setdefault(field, DEFAULT_CONFIG[field])

set_config(o.config_file)
    