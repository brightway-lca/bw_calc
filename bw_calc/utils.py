from .errors import NoArrays
from pathlib import Path
import hashlib
import numpy as np
import zipfile
import json

MAX_SIGNED_32BIT_INT = 2147483647


def load_data_obj(data_obj, check_integrity=True):
    """Load a data obj, provided as either a filepath, a directory path, or a dict."""
    result = {}
    if isinstance(data_obj, dict):
        return data_obj
    elif isinstance(data_obj, (str, Path)):
        if Path(data_obj).is_file() and str(data_obj).endswith(".zip"):
            zf = zipfile.ZipFile(data_obj)
            assert "datapackage.json" in zf.namelist(), "Missing datapackage"
            result = {'datapackage': json.load(zf.open("datapackage.json"))}
            for resource in result['datapackage']['resources']:
                result[resource['path']] = np.load(zf.open(resource['path']), allow_pickle=False)
            return result
        elif Path(data_obj).is_dir():
            dp = Path(data_obj)
            assert (dp / "datapackage.json").is_file(), "Missing datapackage"
            result = {'datapackage': json.load(open(dp / "datapackage.json"))}
            for resource in result['datapackage']['resources']:
                result[resource['path']] = np.load(dp / resource['path'], allow_pickle=False)
            return result
    raise ValueError(f"Can't understand data_obj: '{data_obj}'")


def filter_data_for_matrix(data_objs, matrix_label):
    """Load and concatenate arrays for a given matrix"""
    arrays = [
        load_array(data_obj[resource["path"]])
        for data_obj in data_objs
        for resource in data_obj["datapackage"]["resources"]
        if resource["matrix"] == matrix_label
    ]
    if not arrays:
        raise NoArrays(f"No arrays for '{matrix_label}'")
    return np.hstack(arrays)


def load_array(obj):
    """Load the numpy array if necessary.

    Currently accepts ``str`` filepaths, ``BytesIO``,
     ``numpy.ndarray`` arrays. Creates copies of objects"""
    if isinstance(obj, np.ndarray):
        # we're done here as the object is already a numpy array
        return obj.copy()
    else:
        # treat object as loadable by numpy and try to load it from disk
        return np.load(obj)


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = [
        "uncertainty_type",
        "amount",
        "loc",
        "scale",
        "shape",
        "minimum",
        "maximum",
        "negative",
    ]
    return array[fields].copy()


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, MAX_SIGNED_32BIT_INT)


def md5(filepath, blocksize=65536):
    """Generate MD5 hash for file at `filepath`"""
    hasher = hashlib.md5()
    fo = open(filepath, "rb")
    buf = fo.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fo.read(blocksize)
    return hasher.hexdigest()
