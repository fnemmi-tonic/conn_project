"""
set of companion functions for the toolbox conn
--------------------------------------------------

TO DO : #####Documentation is available in the docstrings and online at
http://nilearn.github.io.

Contents
--------
This module provide a set of useful functions to work with the spm toolbox conn
Submodules
---------
No submodule for the moment
"""

from scipy.io import loadmat
from pathlib import Path
import numpy as np
from glob import glob
from os.path import isdir, isfile
from itertools import chain, combinations
import pandas as pd
from distutils.version import LooseVersion

# Boolean controlling the default globbing technique when using check_niimg
# and the os.path.expanduser usage in CacheMixin.
# Default value it True, set it to False to completely deactivate this
# behavior.
EXPAND_PATH_WILDCARDS = True

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

