from .errors import MalformedFunctionalUnit
import itertools
import datetime
import hashlib
import json
import numpy as np
import os
import tarfile
import tempfile
try:
    import cPickle as pickle
except ImportError:
    import pickle

MAX_INT_32 = 4294967295
MAX_SIGNED_INT_32 = 2147483647


def load_arrays(objs):
    """Load the numpy arrays from list of objects ``objs``.

    Currently accepts ``str`` filepaths, ``BytesIO``,
     ``numpy.ndarray`` arrays. Creates copies of objects"""

    arrays = []
    for obj in objs:
        if isinstance(obj, np.ndarray):
            # we're done here as the object is already a numpy array
            arrays.append(obj.copy())
        else:
            # treat object as loadable by numpy and try to load it from disk
            arrays.append(np.load(obj))

    return np.hstack(arrays)


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc',
              'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, MAX_SIGNED_INT_32)


def md5(filepath, blocksize=65536):
    """Generate MD5 hash for file at `filepath`"""
    hasher = hashlib.md5()
    fo = open(filepath, 'rb')
    buf = fo.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fo.read(blocksize)
    return hasher.hexdigest()
