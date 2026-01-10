import argparse
import glob
import os
import shutil
import sys
from typing import Iterable, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import math
from tqdm import tqdm

import astropy.constants as const
import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table, vstack

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e


# CASA POLARIZATION CORR_TYPE integer codes to labels
_CORR_CODE_TO_NAME = {
    5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
    9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY',
}

def get_corr_label_indices(msname: str):
    """Return (labels_list, label->index dict) from POLARIZATION/CORR_TYPE."""
    t_pol = table(f"{msname}/POLARIZATION", readonly=True)
    corr_types = t_pol.getcell('CORR_TYPE', 0)
    #print(f"found corr_types {corr_types} in {msname}")
    t_pol.close()
    labels = [ _CORR_CODE_TO_NAME.get(int(c), str(c)) for c in corr_types ]
    lbl2idx = {lab: i for i, lab in enumerate(labels)}
    return labels, lbl2idx


def get_channel_lambdas(ms):
    tf = table(f"{ms}/SPECTRAL_WINDOW")
    # get all channel frequencies and convert that to wavelengths
    channel_freqs = tf[0]["CHAN_FREQ"]
    nchan = len(channel_freqs)
    channel_lambdas = const.c.to(u.m/u.s).value / channel_freqs
    tf.close()
    return nchan, channel_freqs, channel_lambdas


def get_time(t):
    vis_time = t.getcol('TIME_CENTROID')
    unique_times = np.unique(vis_time)
    assert np.all(np.abs(np.diff(unique_times) - np.diff(unique_times)[0]) < 1e-2)
    nsub = unique_times.shape[0]
    return nsub, vis_time, unique_times

def get_uvwave_indices(t, channel_lambdas, uvcell_size, N_pixels):    
    #  get the uv positions (in lambdas) for each visibility sample
    uvws = t.getcol('UVW') # in units of metres
    #populate uvw_l grid
    uvws_l = np.ones((uvws.shape[0], uvws.shape[1], len(channel_lambdas))) * uvws[:, :, None]
    # duplicate each uv sample into a size such that we can multiply with the channel lambdas    
    #uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
    
    chan_tiled = np.ones_like(uvws_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
    
    uvws_l = uvws_l / chan_tiled    # get the uvw positions in lambdas


def get_phase_center(msname: str, field_name: str | None = None):
    """
    Returns (ra0_rad, dec0_rad, selected_field_name).
    If field_name is None, uses the first row in FIELD.
    """
    t_field = table(f"{msname}/FIELD", readonly=True)
    names = t_field.getcol("NAME")          # shape: (nfield,)
    phase_dir = t_field.getcol("PHASE_DIR") # shape: (nfield, 2, n_poly) in radians

    idx = 0
    if field_name is not None:
        matches = np.where(names == field_name)[0]
        if matches.size == 0:
            raise ValueError(f"FIELD name '{field_name}' not found in {msname}/FIELD.")
        idx = int(matches[0])

    # Usually polynomial degree n_poly=1. Use the 0th term.
    print(phase_dir[idx])
    ra0_rad  = float(phase_dir[idx][ 0, 0])
    dec0_rad = float(phase_dir[idx][ 0, 1])  #phase_dir[idx]: [[-1.67134852 -0.41158857]]
    return ra0_rad, dec0_rad, str(names[idx])

# --- 3) MS open & chunk count ---
def open_ms(msname: str):
    t_main = table(msname, readonly=True)
    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    return t_main, total_chunks, time_col


def derive_paths(msname: str):
    ms_path = os.path.abspath(msname.rstrip('/'))
    ms_dir  = os.path.dirname(ms_path)
    ms_tag  = os.path.basename(ms_path)
    ms_base = ms_tag[:-3] if ms_tag.endswith('.ms') else ms_tag
    candidates_dir = os.path.join(ms_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    def chunk_prefix_root(start_idx: int) -> str:
        return os.path.join(candidates_dir, f"{ms_base}_chunk_{start_idx:06d}")

    all_prefix_root = os.path.join(candidates_dir, f"{ms_base}_all")

    return ms_base, candidates_dir, chunk_prefix_root, all_prefix_root
