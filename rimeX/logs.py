import logging
import argparse
import rimeX

log_parser = argparse.ArgumentParser(add_help=False)
g = log_parser.add_argument_group("logging")
g.add_argument("--log-file")
m = g.add_mutually_exclusive_group()
m.add_argument("--debug", action='store_const', dest='log_level', const=logging.DEBUG)
m.add_argument("--info", action='store_const', dest='log_level', const=logging.INFO)
m.add_argument("--warning", action='store_const', dest='log_level', const=logging.WARNING)
m.add_argument("--error", action='store_const', dest='log_level', const=logging.ERROR)


# def setup_logger(name=str(Path(__file__).parent), level=logging.INFO, filename=None):
o, _ = log_parser.parse_known_args()
logging.basicConfig(filename=o.log_file)
logger = logging.getLogger(rimeX.__name__)
logger.setLevel(o.log_level or logging.INFO)

del o
del argparse