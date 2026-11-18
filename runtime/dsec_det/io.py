import yaml
import h5py
import numpy as np
import math
import os
from pathlib import Path
import filecmp

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


_H5_FILE_CACHE = {}


def _get_h5_handle(h5file):
    path = str(h5file)
    pid = os.getpid()
    cached = _H5_FILE_CACHE.get(path)

    # Reopen after fork or after an explicit/implicit close.
    if cached is not None:
        cached_pid, handle = cached
        if cached_pid == pid and getattr(handle, "id", None) is not None and handle.id.valid:
            return handle
        try:
            handle.close()
        except Exception:
            pass

    handle = h5py.File(path, 'r')
    _H5_FILE_CACHE[path] = (pid, handle)
    return handle


def close_cached_h5_files():
    for _, handle in _H5_FILE_CACHE.values():
        try:
            handle.close()
        except Exception:
            pass
    _H5_FILE_CACHE.clear()


# from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
class dircmp(filecmp.dircmp):
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """
    def phase3(self):
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files,
                                 shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp

def compare_dirs(dir1: Path, dir2: Path):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if (compared.left_only or compared.right_only or compared.diff_files
        or compared.funny_files):
        return False
    for subdir in compared.common_dirs:
        if not compare_dirs(dir1 / subdir, dir2 / subdir):
            return False
    return True

def _extract_from_h5_by_index(filehandle, ev_start_idx: int, ev_end_idx: int):
    events = filehandle['events']
    x = events['x']
    y = events['y']
    p = events['p']
    t = events['t']

    x_new = x[ev_start_idx:ev_end_idx]
    y_new = y[ev_start_idx:ev_end_idx]
    p_new = p[ev_start_idx:ev_end_idx]
    t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

    output = {
        'p': p_new,
        't': t_new,
        'x': x_new,
        'y': y_new,
    }
    return output


def get_num_events(h5file):
    h5f = _get_h5_handle(h5file)
    return len(h5f['events/t'])

def extract_from_h5_by_index(h5file, ev_start_idx: int, ev_end_idx: int):
    h5f = _get_h5_handle(h5file)
    return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)

def extract_from_h5_by_timewindow(h5file, t_min_us: int, t_max_us: int):
    h5f = _get_h5_handle(h5file)
    ms2idx = np.asarray(h5f['ms_to_idx'], dtype='int64')
    t_offset = h5f['t_offset'][()]

    events = h5f['events']
    t = events['t']

    t_ev_start_us = t_min_us - t_offset
    #assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
    t_ev_start_ms = t_ev_start_us // 1000
    ms2idx_start_idx = t_ev_start_ms
    ev_start_idx = ms2idx[ms2idx_start_idx]

    t_ev_end_us = t_max_us - t_offset
    #assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
    t_ev_end_ms = math.floor(t_ev_end_us / 1000)
    ms2idx_end_idx = np.clip(t_ev_end_ms, 0, len(ms2idx)-1)
    ev_end_idx = ms2idx[ms2idx_end_idx]

    return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)

def h5_file_to_dict(h5_file: Path):
    fh = _get_h5_handle(h5_file)
    return {k: fh[k][()] for k in fh.keys()}

def yaml_file_to_dict(yaml_file: Path):
    with yaml_file.open() as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)
