from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Any, Optional
import numpy as np

@dataclass
class Config:
    msname: str
    npix_x: int
    npix_y: int
    pix_rad: float
    ra0_rad: float
    dec0_rad: float
    epsilon: float
    do_wgridding: bool
    nthreads: int
    verbosity: int
    corr_mode: str
    basis: str
    single_pol: str
    data_column: str
    enable_var: bool
    enable_var_chunk: bool
    enable_boxcar: bool
    var_threshold: float
    var_keep_k: Optional[int]
    var_clip_sigma: float
    var_nms_radius: int
    boxcar_widths: List[int]
    boxcar_threshold: float
    do_plot: bool
    ms_base: str
    candidates_dir: str
    chunk_prefix_root: Any
    all_prefix_root: str
    save_var_lightcurves: bool
    save_var_snippets: bool
    save_box_lightcurves: bool
    save_box_snippets: bool

@dataclass
class WelfordState:
    count: np.ndarray
    mean:  np.ndarray
    M2:    np.ndarray
