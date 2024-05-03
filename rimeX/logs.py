import logging
import argparse
import rimeX

log_parser = argparse.ArgumentParser(add_help=False)
g = log_parser.add_argument_group("logging")
g.add_argument("--log-file")
g.add_argument("--debug", action='store_const', dest='log_level', const=logging.DEBUG)
g.add_argument("--info", action='store_const', dest='log_level', const=logging.INFO)
g.add_argument("--warning", action='store_const', dest='log_level', const=logging.WARNING)
g.add_argument("--error", action='store_const', dest='log_level', const=logging.ERROR)

def init_logger(cmd=None):
    o, _ = log_parser.parse_known_args(cmd)
    setup_logger(o)

logger = logging.getLogger(rimeX.__name__)
formatter = logging.Formatter('[%(asctime)s | %(name)s | %(levelname)s] %(message)s', "%H:%M:%S")
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(formatter)

def setup_logger(o):
    # source: https://stackoverflow.com/a/59705351/2192272
    logger.handlers.clear()
    if o.log_file:
        handler = logging.FileHandler(o.log_file)
        handler.setFormatter(formatter)
    else:
        handler = streamhandler
    logger.addHandler(handler)
    logger.setLevel(o.log_level or logging.INFO)

init_logger([])


del argparse