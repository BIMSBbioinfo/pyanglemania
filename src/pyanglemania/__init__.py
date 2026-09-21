from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import datasets, preprocessing
from . import preprocessing as pp

try:
    __version__ = version("pyanglemania")
except PackageNotFoundError:  # not installed (e.g. run straight from a checkout)
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "datasets", "preprocessing", "pp"]
