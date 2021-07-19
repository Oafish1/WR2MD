#https://stackoverflow.com/a/1057534
import glob
from os.path import basename, dirname, isfile, join

modules = glob.glob(join(dirname(__file__), "*"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__.py')]
__all__ += [basename(d) for d in modules if not isfile(d) and not d.endswith('__')]

from . import *
