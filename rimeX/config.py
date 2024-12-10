import os
import argparse
from pathlib import Path
from flatdict import FlatDict
import toml

import rimeX
from rimeX.logs import logger, setup_logger

def load_config(file):
    with open(file) as f:
        cfg = toml.load(f)
    return FlatDict(cfg, delimiter=".")


DEFAULT_CONFIG_FILE = Path(__file__).parent/"config.toml"
DEFAULT_CONFIG = load_config(DEFAULT_CONFIG_FILE)
CONFIG = DEFAULT_CONFIG.copy()

# Sources:
# https://stackoverflow.com/questions/305647/appropriate-location-for-my-applications-cache-on-windows
# https://stackoverflow.com/questions/1325581/how-do-i-check-if-im-running-on-windows-in-python
if os.name == 'nt':
    APPDATA = Path(os.getenv("ALLUSERSPROFILE")) / rimeX.__name__
    CACHE_FOLDER = APPDATA / "data_download"
    GLOBAL_CONFIG_FILE = APPDATA / (rimeX.__name__ + ".toml")

# Source: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
else:
    _HOME = os.getenv("HOME")
    _CACHE_FOLDER_SYSTEM = os.getenv("XDG_CACHE_HOME", os.path.join(_HOME, ".cache"))
    CACHE_FOLDER = Path(_CACHE_FOLDER_SYSTEM) / rimeX.__name__
    _CONFIG_FOLDER_SYSTEM = os.getenv("XDG_CONFIG_HOME", os.path.join(_HOME, ".config"))
    GLOBAL_CONFIG_FILE = Path(_CONFIG_FOLDER_SYSTEM) / (rimeX.__name__ + ".toml")

def search_default_config():
    # seach current working directory
    candidates = ['rimeX.toml', 'rime.toml']
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
    global CONFIG
    CONFIG = load_config(file_path)

    # update undefined fields with defaults
    for field, default_value in DEFAULT_CONFIG.items():
        CONFIG.setdefault(field, default_value)

    # also extend the indicators list the ISIMIP variables
    CONFIG.setdefault("indicators", CONFIG["isimip.variables"] + sorted(CONFIG.get("indicator", {})))

    # update some variables
    if type(CONFIG["isimip.simulation_round"]) is str:
        CONFIG["isimip.simulation_round"] = [CONFIG["isimip.simulation_round"]]

    # type check here?

set_config(o.config_file)


def reset_config():
    file_path = search_default_config()
    set_config(file_path)

def get_outputpath(relpath):
    return Path(CONFIG.get("output_folder", "output")) / relpath

def main():
    """show configuration file"""
    parser = argparse.ArgumentParser(parents=[config_parser])
    o = parser.parse_args()
    print(toml.dumps(CONFIG.as_dict()))
    parser.exit(0)


if __name__ == "__main__":
    main()

