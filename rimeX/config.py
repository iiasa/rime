import argparse
from pathlib import Path
from flatdict import FlatDict
import toml

from rimeX.logs import logger

def load_config(file):
    with open(file) as f:
        cfg = toml.load(f)
    return FlatDict(cfg, delimiter=".")

DEFAULT_CONFIG_FILE = Path(__file__).parent/"config.toml"
DEFAULT_CONFIG = load_config(DEFAULT_CONFIG_FILE)
CONFIG = DEFAULT_CONFIG.copy()

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

set_config(o.config_file)


def main():
    """show configuration file"""
    parser = argparse.ArgumentParser(parents=[config_parser])
    o = parser.parse_args()
    print(toml.dumps(CONFIG.as_dict()))
    parser.exit(0)


if __name__ == "__main__":
    main()
    