import argparse
import glob
import os
import shutil
import sys
from tqdm import tqdm
import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from numba import njit, prange

from casacore.tables import table
from typing import Iterable, Tuple, List, Dict, Any, Optional


# CASA POLARIZATION CORR_TYPE integer codes to labels
_CORR_CODE_TO_NAME = {
    5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
    9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY',
}


RCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True)
class World:
    
    """
    Holds metadata necessary for pixel->(l,m)->RA/Dec conversion.

    - ra0_rad / dec0_rad: phase center (radians).
    - pixel_scale_arcsec: cellsize (arcsec per pixel) used by imaging.
    - x0 / y0: reference pixel coordinates corresponding to phase center.
    - field_name: optional FIELD name from MS.
    - x_toward_east / y_toward_north: axis orientation flags.
    """

    ra0_rad: float
    dec0_rad: float
    pixel_scale_arcsec: float
    x0: float
    y0: float
    field_name: str | None = None
    x_toward_east: bool = True
    y_toward_north: bool = True

    @property
    def pixel_scale_rad(self) -> float:
        return self.pixel_scale_arcsec * ARCSEC_TO_RAD

def lm_from_pixel_xy(x: np.ndarray, y: np.ndarray, world: World) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel centroids (x,y) -> (l,m) in radians relative to phase center.
    Vectorized for arrays.
    """
    
    dx_pix = np.asarray(x, dtype=np.float64) - world.x0
    dy_pix = np.asarray(y, dtype=np.float64) - world.y0

    sx = +1.0 if world.x_toward_east  else -1.0
    sy = +1.0 if world.y_toward_north else -1.0

    l = sx * dx_pix * world.pixel_scale_rad
    m = sy * dy_pix * world.pixel_scale_rad
    return l, m    
    
def radec_from_lm(l: np.ndarray, m: np.ndarray, world: World) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized (l,m) -> (RA, Dec) in radians using TAN/gnomic
    """

    ra0, dec0 = world.ra0_rad, world.dec0_rad

    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    # Direction cosine n
    lm2 = l*l + m*m
    n = np.sqrt(np.clip(1.0 - lm2, 0.0, None))

    sin_dec = m * np.cos(dec0) + n * np.sin(dec0)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    denom = (n * np.cos(dec0) - m * np.sin(dec0))
    d_ra = np.arctan2(l, denom)
    ra = (ra0 + d_ra) % (2.0 * math.pi)

    return ra, dec

    

# ============================
# NUMBA-ACCELERATED KERNELS
# ============================

@njit
def _moving_sum_from_csum(csum: np.ndarray, w: int) -> np.ndarray:
    """
    csum: (T+1, Ny, Nx) cumulative sum along time.
    returns S: (T-w+1, Ny, Nx) windowed sum for width w.
    """
    T_plus, Ny, Nx = csum.shape
    T = T_plus - 1
    T_eff = T - w + 1
    out = np.empty((T_eff, Ny, Nx), dtype=csum.dtype)
    for t0 in range(T_eff):
        out[t0, :, :] = csum[t0 + w, :, :] - csum[t0, :, :]
    return out


@njit
def _median_1d(arr: np.ndarray, n: int) -> float:
    """
    Median over arr[:n] (n valid values). arr is modified (sorted).
    Assumes n >= 1.
    """
    tmp = arr[:n].copy()
    tmp.sort()
    mid = n // 2
    if (n % 2) == 1:
        return tmp[mid]
    else:
        return 0.5 * (tmp[mid - 1] + tmp[mid])


@njit
def _mad_std_2d_per_time(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Robust spatial std per time using 1.4826 * MAD across (y,x).
    S: (T_eff, Ny, Nx)
    mask: (Ny, Nx) boolean; True means valid pixel
    Returns sigma_w: (T_eff,)
    """
    T_eff, Ny, Nx = S.shape
    sigma_w = np.empty(T_eff, dtype=S.dtype)

    # scratch buffers
    buf_vals = np.empty(Ny * Nx, dtype=S.dtype)
    buf_dev = np.empty(Ny * Nx, dtype=S.dtype)

    for t0 in range(T_eff):
        n = 0
        # collect masked values
        for y in range(Ny):
            for x in range(Nx):
                if mask[y, x]:
                    v = S[t0, y, x]
                    if np.isfinite(v):
                        buf_vals[n] = v
                        n += 1
        if n == 0:
            sigma_w[t0] = 1.0  # fallback
            continue

        med = _median_1d(buf_vals, n)

        # MAD
        m = 0
        for i in range(n):
            d = abs(buf_vals[i] - med)
            if np.isfinite(d):
                buf_dev[m] = d
                m += 1
        if m == 0:
            sigma_w[t0] = 1.0
            continue
        mad = _median_1d(buf_dev, m)
        sigma_w[t0] = 1.4826 * mad if mad > 0 else 1.0
    return sigma_w


@njit
def _clipped_rms_2d_per_time(S: np.ndarray, mask: np.ndarray, sigma: float, max_iter: int = 5) -> np.ndarray:
    """
    Iterative sigma-clipped RMS per time across (y,x).
    S: (T_eff, Ny, Nx)
    mask: (Ny, Nx)
    Returns sigma_w: (T_eff,)
    """
    T_eff, Ny, Nx = S.shape
    sigma_w = np.empty(T_eff, dtype=S.dtype)

    buf_vals = np.empty(Ny * Nx, dtype=S.dtype)

    for t0 in range(T_eff):
        n = 0
        for y in range(Ny):
            for x in range(Nx):
                if mask[y, x]:
                    v = S[t0, y, x]
                    if np.isfinite(v):
                        buf_vals[n] = v
                        n += 1
        if n == 0:
            sigma_w[t0] = 1.0
            continue

        # robust init via median/MAD
        med = _median_1d(buf_vals, n)
        # MAD
        m = 0
        for i in range(n):
            d = abs(buf_vals[i] - med)
            if np.isfinite(d):
                buf_vals[m] = d  # reuse buffer
                m += 1
        mad = _median_1d(buf_vals, m) if m > 0 else 0.0
        std = 1.4826 * mad if mad > 0 else 0.0
        mu = med

        # iterative clipping
        for _ in range(max_iter):
            if not np.isfinite(std) or std <= 0:
                break
            keep_count = 0
            for i in range(n):
                if abs(buf_vals[i] - mu) <= sigma * std:
                    buf_vals[keep_count] = buf_vals[i]
                    keep_count += 1
            if keep_count == 0:
                break
            # recompute mean/std on kept
            # mean
            s = 0.0
            for i in range(keep_count):
                s += buf_vals[i]
            mu = s / keep_count
            # std (ddof=1)
            ss = 0.0
            for i in range(keep_count):
                d = buf_vals[i] - mu
                ss += d * d
            std = np.sqrt(ss / max(1, keep_count - 1))
        sigma_w[t0] = std if (np.isfinite(std) and std > 0) else 1.0
    return sigma_w


@njit(parallel=True)
def _temporal_std_snr(data: np.ndarray, csum: np.ndarray, csum2: np.ndarray, w: int) -> np.ndarray:
    """
    Per-pixel temporal variance over window width w, SNR computed as s / (std * sqrt(w)).
    data: (T, Ny, Nx) float64
    csum: (T+1, Ny, Nx)
    csum2: (T+1, Ny, Nx)
    returns snr: (T-w+1, Ny, Nx)
    """
    T, Ny, Nx = data.shape
    T_eff = T - w + 1
    snr = np.zeros((T_eff, Ny, Nx), dtype=np.float64)
    for t0 in prange(T_eff):
        # windowed sums
        s = csum[t0 + w, :, :] - csum[t0, :, :]
        s2 = csum2[t0 + w, :, :] - csum2[t0, :, :]
        # mean & var
        mean = s / w
        var = s2 / w - mean * mean
        # clip negatives
        for y in range(Ny):
            for x in range(Nx):
                v = var[y, x]
                if v < 0.0:
                    v = 0.0
                std = np.sqrt(v)
                denom = std * np.sqrt(w)
                snr[t0, y, x] = (s[y, x] / denom) if denom > 0 else 0.0
    return snr


@njit
def _max_filter_2d(src: np.ndarray, radius: int) -> np.ndarray:
    """
    Simple 2D max filter with square window (size 2*radius+1).
    Pads by nearest behavior (clamped indices).
    """
    Ny, Nx = src.shape
    k = 2 * radius + 1
    out = np.empty((Ny, Nx), dtype=src.dtype)
    for y in range(Ny):
        for x in range(Nx):
            y0 = max(0, y - radius)
            y1 = min(Ny - 1, y + radius)
            x0 = max(0, x - radius)
            x1 = min(Nx - 1, x + radius)
            m = -np.inf
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    v = src[yy, xx]
                    if v > m:
                        m = v
            out[y, x] = m
    return out



def _get_corr_label_indices(msname: str):
    """Return (labels_list, label->index dict) from POLARIZATION/CORR_TYPE."""
    t_pol = table(f"{msname}/POLARIZATION", readonly=True)
    corr_types = t_pol.getcell('CORR_TYPE', 0)
    #print(f"found corr_types {corr_types} in {msname}")
    t_pol.close()
    labels = [ _CORR_CODE_TO_NAME.get(int(c), str(c)) for c in corr_types ]
    lbl2idx = {lab: i for i, lab in enumerate(labels)}
    return labels, lbl2idx

def parse_args():
    parser = argparse.ArgumentParser(description="Run CASA applycal on MS files for specified beams (SBID-aware).")
    parser.add_argument("--msname", required=True, help="measurement set name")
    parser.add_argument("--cube-length", default=1000, help="maximum length of output cube chunk")
    parser.add_argument("--img-interval", default=1, help="imaging interval in units of samples")    
    parser.add_argument("--channels-out", default=1, help="number of channels to write out")
    parser.add_argument("--do-uv", default=1, help="also save uv grid cube")
    return parser.parse_args()

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
    ra0_rad  = float(phase_dir[idx, 0, 0])
    dec0_rad = float(phase_dir[idx, 1, 0])
    return ra0_rad, dec0_rad, str(names[idx])

    

def image_time_samples(
    msname: str,
    *,
    start_time_idx: int | None = None,
    end_time_idx: int | None = None,
    data_column: str = 'DATA',
    corr_mode: str = 'average',  # 'average' | 'stokesI' | 'single'
    basis: str = 'auto',         # for stokesI: 'auto' | 'linear' | 'circular'
    single_pol: str = 'XX',      # used when corr_mode='single'
    average_correlations: bool = True,
    corr_index: int | None = None,
    use_weight_spectrum: bool = True,
    npix_x: int = 384,
    npix_y: int = 384,
    pixsize_x: float = 22.0/206265.0,
    pixsize_y: float = 22.0/206265.0,
    epsilon: float = 1e-6,
    do_wgridding: bool = True,
    nthreads: int = 0,
    verbosity: int = 0,
    flip_u: bool = False,
    flip_v: bool = False,
    flip_w: bool = False,
    divide_by_n: bool = True,
    sigma_min: float = 1.1,
    sigma_max: float = 2.6,
    center_x: float = 0.0,
    center_y: float = 0.0,
    allow_nshift: bool = True,
    double_precision_accumulation: bool = False,
    do_plot: bool = False,
):
    """
    Iterate over time samples of a Measurement Set and grid visibilities
    into dirty images using ducc0.wgridder.vis2dirty.

    Parameters
    ----------
    start_time_idx : int | None
        0-based index of the first time chunk to process (inclusive). If None, start at 0.
    end_time_idx : int | None
        0-based index of the last time chunk to process (inclusive). If None, process until the end.

    Returns
    -------
    list[tuple[float, np.ndarray]]
        A list of (time_value, dirty_image) for each processed time sample.
    """

    try:
        import ducc0
    except Exception as e:
        raise RuntimeError('ducc0 is required for image_time_samples()') from e

    t_main = table(msname, readonly=True)
    colnames = set(t_main.colnames())
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in colnames else 'TIME'

    # Frequencies
    t_spw = table(f"{msname}/SPECTRAL_WINDOW", readonly=True)
    n_spw = t_spw.nrows()
    if n_spw != 1:
        raise ValueError(f"image_time_samples() currently supports a single SPW; found {n_spw}")
    chan_freq = t_spw.getcell('CHAN_FREQ', 0)  # [nchan] Hz
    t_spw.close()

    #set up iterator over table, grouped by common timestamps
    it = t_main.iter([time_col], sort=True)
    #declare results list as list of tuple of float (for timestamp) and ndarray (for image)
    #results: list[tuple[float, np.ndarray]] = []

    
    nsub, vis_time, all_times = get_time(t_main)
    #pre initialise img cube in mem and fill as we go
    # the number of time samples should be the min of (N - start_idx, end_idx - start_idx, N) where N is no. of time samples

    start_idx = 0 if start_time_idx is None else int(start_time_idx)
    end_idx   = (nsub - 1) if end_time_idx is None else int(end_time_idx)
    if start_idx < 0 or end_idx < start_idx or end_idx >= nsub:
        raise ValueError("Invalid start/end time indices")

    times = all_times[start_idx:end_idx+1]
    nt_window = end_idx - start_idx + 1
    
    cube = np.empty((nt_window, npix_y, npix_x))#, dtype=dtype_img)
    
    cube_idx = 0
    labels, lbl2idx = _get_corr_label_indices(msname)
    
    for t_chunk_idx, t_chunk in enumerate(tqdm(it)):
        # Apply start/end time-chunk windowing
        if start_time_idx is not None and t_chunk_idx < start_time_idx:
            continue
        if end_time_idx is not None and t_chunk_idx > end_time_idx:
            break

        times_ = t_chunk.getcol(time_col)
        time_val = float(times_[0])

        uvw   = t_chunk.getcol('UVW')
        data  = t_chunk.getcol(data_column)   # [nrow, nchan, ncorr]
        flags = t_chunk.getcol('FLAG')        # [nrow, nchan, ncorr]
        flag_row = t_chunk.getcol('FLAG_ROW') if 'FLAG_ROW' in set(t_chunk.colnames()) else None
        
        # Weights
        if use_weight_spectrum and 'WEIGHT_SPECTRUM' in set(t_chunk.colnames()):
            wgt = t_chunk.getcol('WEIGHT_SPECTRUM')
        else:
            wgt_row = t_chunk.getcol('WEIGHT')
            wgt = np.broadcast_to(wgt_row[:, None, :], data.shape)

        # Apply flags -> zero weights
        good = ~flags
        if flag_row is not None:
            good &= (~flag_row[:, None, None])
        wgt = np.where(good, wgt, 0.0)

        # Correlation collapse
        if corr_mode == 'average':
            if average_correlations:
                wsum = wgt.sum(axis=2)
                with np.errstate(invalid='ignore', divide='ignore'):
                    vis = (data * wgt).sum(axis=2) / np.where(wsum > 0.0, wsum, np.nan)
                vis = np.nan_to_num(vis, nan=0.0)
                wgt_2d = wsum
            else:
                if corr_index is None:
                    corr_index = 0
                vis    = data[:, :, corr_index]
                wgt_2d = wgt[:, :, corr_index]
        elif corr_mode == 'single':
            if single_pol not in lbl2idx:
                raise ValueError(f"Requested single_pol='{single_pol}' not present in MS correlations: {labels}")
            ci = lbl2idx[single_pol]
            vis    = data[:, :, ci]
            wgt_2d = wgt[:, :, ci]
        elif corr_mode == 'stokesI':
            have_linear   = ('XX' in lbl2idx) and ('YY' in lbl2idx)
            have_circular = ('RR' in lbl2idx) and ('LL' in lbl2idx)
            use_linear = False
            use_circ   = False
            if basis == 'linear':
                use_linear = have_linear
                if not use_linear:
                    raise ValueError("basis='linear' requested but XX/YY not found in MS correlations")
            elif basis == 'circular':
                use_circ = have_circular
                if not use_circ:
                    raise ValueError("basis='circular' requested but RR/LL not found in MS correlations")
            else:
                if have_linear:
                    use_linear = True
                elif have_circular:
                    use_circ = True
                else:
                    raise ValueError("Cannot form Stokes I: XX/YY or RR/LL not present in MS correlations")
            if use_linear:
                i1, i2 = lbl2idx['XX'], lbl2idx['YY']
            else:
                i1, i2 = lbl2idx['RR'], lbl2idx['LL']
            v1, w1 = data[:, :, i1], wgt[:, :, i1]
            v2, w2 = data[:, :, i2], wgt[:, :, i2]
            present1 = (w1 > 0.0)
            present2 = (w2 > 0.0)
            n_valid  = present1.astype(np.int32) + present2.astype(np.int32)
            sum_vis = np.zeros_like(v1)
            sum_vis += np.where(present1, v1, 0.0)
            sum_vis += np.where(present2, v2, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                vis = sum_vis / np.where(n_valid > 0, n_valid, np.nan)
            vis = np.nan_to_num(vis, nan=0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                w_two = 4.0 / (np.where(present1, 1.0 / w1, 0.0) + np.where(present2, 1.0 / w2, 0.0))
            w_one = np.where(present1 & (~present2), w1, 0.0) + np.where((~present1) & present2, w2, 0.0)
            wgt_2d = np.where(n_valid == 2, np.nan_to_num(w_two, nan=0.0, posinf=0.0, neginf=0.0), w_one)
        else:
            raise ValueError(f"Unknown corr_mode='{corr_mode}'. Use 'average', 'stokesI', or 'single'.")

        if uvw.shape[0] != vis.shape[0]:
            raise ValueError('Row count mismatch between UVW and VIS')
        if chan_freq.shape[0] != vis.shape[1]:
            raise ValueError('Channel count mismatch between CHAN_FREQ and VIS')

        #print(np.max(wgt_2d), np.median(wgt_2d), np.min(wgt_2d))
        wgt_2d /= np.max(wgt_2d)
        
        dirty = ducc0.wgridder.vis2dirty(
            uvw=uvw,
            freq=chan_freq,
            vis=vis,
            wgt=wgt_2d,
            npix_x=npix_x,
            npix_y=npix_y,
            pixsize_x=pixsize_x,
            pixsize_y=pixsize_y,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            verbosity=verbosity,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            divide_by_n=divide_by_n,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            center_x=center_x,
            center_y=center_y,
            allow_nshift=allow_nshift,
            double_precision_accumulation=double_precision_accumulation,
        )


        n_valid = int(np.count_nonzero(wgt_2d)) #np.sum(wgt_2d/np.max(wgt_2d)) #
        if n_valid > 0:
            #I think divide_by_n should be dealing with this already but whatever
            dirty = dirty / n_valid

        
        # if t_chunk_idx == 10:
        #     print("adding fake transient")
        #     #dirty[198:202,198:202] = 100.0
        #     import matplotlib.pyplot as plt
        #     plt.imshow(dirty, aspect='auto', interpolation='none', origin='lower', cmap='gray')
        #     plt.colorbar()
        #     plt.savefig(f'img_{cube_idx:04d}.png', dpi=300)
        #     plt.close()
            
        cube[cube_idx, :, :] = dirty
        cube_idx += 1

        if do_plot:
            import matplotlib.pyplot as plt
            plt.imshow(dirty, aspect='auto', interpolation='none', origin='lower', cmap='gray')
            plt.colorbar()            
            plt.savefig(f"img_{cube_idx:04d}.png", dpi=300)
            sys.exit()
            #plt.show()
        #results.append((time_val, dirty))
        
    #it.close()
    t_main.close()
    
    results = (times, cube)    
    return results



def boxcar_search_time(
    times: np.ndarray,
    cube: np.ndarray,  # (T, Ny, Nx)
    widths: Iterable[float],
    *,
    widths_in_seconds: bool = False,
    threshold_sigma: float = 5.0,
    return_snr_cubes: bool = False,
    keep_top_k: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    subtract_mean_per_pixel: bool = False,
    std_mode: str = "spatial_per_window",  # "spatial_per_window" | "temporal_per_pixel"
    spatial_estimator: str = "clipped_rms",  # "mad" | "clipped_rms"
    clip_sigma: float = 3.0,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[int, np.ndarray]]]:
    """
    Numba-accelerated SNR computations.
    """
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")

    # Mask
    if valid_mask is None:
        valid_mask = np.ones((Ny, Nx), dtype=bool)
    elif valid_mask.shape != (Ny, Nx):
        raise ValueError("valid_mask must be shape (Ny, Nx)")

    # High-pass in time per pixel (optional)
    data = cube.astype(np.float64, copy=False)
    if subtract_mean_per_pixel:
        mean_map = data.mean(axis=0, keepdims=True)
        data = data - mean_map

    # Convert widths
    widths_samples: List[int] = []
    if widths_in_seconds:
        dt = np.median(np.diff(times))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid dt; cannot convert seconds to samples.")
        for w_sec in widths:
            widths_samples.append(max(1, int(round(w_sec / dt))))
    else:
        for w in widths:
            widths_samples.append(max(1, int(w)))

    detections: List[Dict[str, Any]] = []
    snr_cubes: Optional[Dict[int, np.ndarray]] = {} if return_snr_cubes else None

    # cumulative sums for fast windowed sums
    csum = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum[1:] = np.cumsum(data, axis=0)
    # Only needed for temporal std mode
    csum2 = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum2[1:] = np.cumsum(data * data, axis=0)

    mask3 = valid_mask[None, :, :]  # broadcast helper

    for w in widths_samples:
        if w > T:
            continue

        if std_mode == "spatial_per_window":
            S = _moving_sum_from_csum(csum, w)  # (T_eff, Ny, Nx)
            T_eff = S.shape[0]

            # Compute a spatial sigma per time window
            if spatial_estimator == "mad":
                sigma_w = _mad_std_2d_per_time(S, valid_mask)  # (T_eff,)
            elif spatial_estimator == "clipped_rms":
                sigma_w = _clipped_rms_2d_per_time(S, valid_mask, clip_sigma)
            else:
                raise ValueError(f"Unknown spatial_estimator='{spatial_estimator}'")

            # SNR: S / sigma_w[t]
            snr = np.empty_like(S)
            for t0 in range(T_eff):
                s = sigma_w[t0]
                if s <= 0 or not np.isfinite(s):
                    s = 1.0
                snr[t0, :, :] = S[t0, :, :] / s

        elif std_mode == "temporal_per_pixel":
            snr = _temporal_std_snr(data, csum, csum2, w)  # (T_eff, Ny, Nx)
            T_eff = snr.shape[0]
        else:
            raise ValueError(f"Unknown std_mode='{std_mode}'")

        # apply valid mask (additional gating)
        snr = np.where(mask3, snr, 0.0)

        # optionally export SNR cube
        if return_snr_cubes:
            snr_cubes[w] = snr.astype(np.float32, copy=False)

        # threshold
        hits = np.where(snr >= threshold_sigma)
        t0_idx, ys, xs = hits
        if t0_idx.size == 0:
            continue

        # time centers for window [t0, t0+w)
        centers = times[t0_idx + (w // 2)]

        # Top-K per width
        if keep_top_k is not None and t0_idx.size > keep_top_k:
            order = np.argsort(snr[t0_idx, ys, xs])[::-1]
            sel = order[:keep_top_k]
            t0_idx, ys, xs, centers = t0_idx[sel], ys[sel], xs[sel], centers[sel]

        # pack detections
        for i in range(t0_idx.size):
            t0 = int(t0_idx[i])
            t1 = t0 + w  # exclusive, safe: t1 <= T
            time_start = float(times[t0])
            time_end = float(times[t1 - 1])
            time_center = 0.5 * (time_start + time_end)

            # nearest index to time_center, constrained to [t0, t1-1]
            k = np.searchsorted(times[t0:t1], time_center)
            if k == 0:
                center_idx = t0
            elif k >= (t1 - t0):
                center_idx = t1 - 1
            else:
                left = t0 + (k - 1)
                right = t0 + k
                center_idx = left if abs(times[left] - time_center) <= abs(times[right] - time_center) else right

            det = {
                "time_start": time_start,
                "time_end": time_end,
                "time_center": float(time_center),
                "t0_idx": t0,
                "t1_idx_excl": t1,
                "center_idx": int(center_idx),
                "y": int(ys[i]),
                "x": int(xs[i]),
                "width_samples": int(w),
                "snr": float(snr[t0, ys[i], xs[i]]),
                # value_sum is useful only for spatial_per_window
                "value_sum": float(snr[t0, ys[i], xs[i]]) if std_mode != "spatial_per_window" else float(
                    _moving_sum_from_csum(csum, w)[t0, ys[i], xs[i]]
                ),
            }
            detections.append(det)

    return detections, snr_cubes

def nms_snr_maps_per_width(
    snr_cubes: Dict[int, np.ndarray],  # width_samples -> SNR cube (T_eff, Ny, Nx)
    times: np.ndarray,
    *,
    threshold_sigma: float = 5.0,
    spatial_radius: int = 3,
    time_radius: int = 0,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Spatial (and optional temporal) NMS using compiled 2D max-filter for peaks.
    """
    detections_by_width: Dict[int, List[Dict[str, Any]]] = {}

    for w, snr_w in snr_cubes.items():
        T_eff, Ny, Nx = snr_w.shape
        if valid_mask is not None and valid_mask.shape != (Ny, Nx):
            raise ValueError("valid_mask must match (Ny, Nx) of SNR cubes")

        detections: List[Dict[str, Any]] = []

        # temporal occupancy (optional)
        if time_radius > 0:
            occ_time = [np.ones((Ny, Nx), dtype=bool) for _ in range(T_eff)]
        else:
            occ_time = None

        for t0 in range(T_eff):
            # copy slice
            snr_slice = snr_w[t0].copy()

            # apply mask & threshold first
            if valid_mask is not None:
                snr_slice = np.where(valid_mask, snr_slice, -np.inf)
            snr_slice = np.where(snr_slice >= threshold_sigma, snr_slice, -np.inf)

            # temporal occupancy gate
            if occ_time is not None:
                gate = np.ones((Ny, Nx), dtype=bool)
                for dt in range(-time_radius, time_radius + 1):
                    t2 = t0 + dt
                    if 0 <= t2 < T_eff:
                        gate &= occ_time[t2]
                snr_slice = np.where(gate, snr_slice, -np.inf)

            # local maxima
            local_max = _max_filter_2d(snr_slice, spatial_radius)
            peaks_mask = (snr_slice >= local_max) & np.isfinite(snr_slice)
            ys, xs = np.where(peaks_mask)
            if ys.size == 0:
                continue

            # sort by SNR descending
            snr_vals = snr_slice[ys, xs]
            order = np.argsort(snr_vals)[::-1]
            ys, xs, snr_vals = ys[order], xs[order], snr_vals[order]

            # accept peaks, suppress neighbors
            for y, x, s in zip(ys, xs, snr_vals):
                y0 = max(0, y - spatial_radius)
                y1 = min(Ny, y + spatial_radius + 1)
                x0 = max(0, x - spatial_radius)
                x1 = min(Nx, x + spatial_radius + 1)

                t1 = t0 + w  # exclusive
                time_start = float(times[t0])
                time_end = float(times[t1 - 1])
                time_center = 0.5 * (time_start + time_end)

                k = np.searchsorted(times[t0:t1], time_center)
                if k == 0:
                    center_idx = t0
                elif k >= (t1 - t0):
                    center_idx = t1 - 1
                else:
                    left = t0 + (k - 1)
                    right = t0 + k
                    center_idx = left if abs(times[left] - time_center) <= abs(times[right] - time_center) else right

                det = {
                    "y": int(y),
                    "x": int(x),
                    "snr": float(s),
                    "width_samples": int(w),
                    "t0_idx": int(t0),
                    "t1_idx_excl": int(t1),
                    "center_idx": int(center_idx),
                    "time_start": time_start,
                    "time_end": time_end,
                    "time_center": float(time_center),
                }
                detections.append(det)

                # suppress neighborhood in current time
                snr_w[t0, y0:y1, x0:x1] = -np.inf
                if occ_time is not None:
                    for dt in range(-time_radius, time_radius + 1):
                        t2 = t0 + dt
                        if 0 <= t2 < T_eff:
                            snr_w[t2, y0:y1, x0:x1] = -np.inf
                            occ_time[t2][y0:y1, x0:x1] = False

        detections_by_width[w] = detections

    return detections_by_width

def group_filter_across_widths(
        detections_by_width: Dict[int, List[Dict[str, Any]]],
        times: np.ndarray,
        *,
        spatial_radius: int = 3,
        time_radius: int = 0,
        policy: str = "max_snr",
        max_per_time_group: Optional[int] = None,
        ny_nx: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:



        # Flatten by time center
    by_time: Dict[int, List[Dict[str, Any]]] = {}
    for w, dets in detections_by_width.items():
        for d in dets:
            ci = int(d["center_idx"])
            by_time.setdefault(ci, []).append(d)

    # If temporal NMS is requested, prepare a global occupancy structure keyed by center_idx
    occ_by_center: Dict[int, np.ndarray] = {}

    # If Ny,Nx not provided, infer from detections
    if ny_nx is None:
        max_y = -1
        max_x = -1
        for dets in by_time.values():
            for d in dets:
                if d["y"] > max_y: max_y = d["y"]
                if d["x"] > max_x: max_x = d["x"]
        Ny = max_y + 1 if max_y >= 0 else 1
        Nx = max_x + 1 if max_x >= 0 else 1
    else:
        Ny, Nx = ny_nx

    final: List[Dict[str, Any]] = []

    # Helper to get sorted order per policy
    def sort_dets(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if policy == "max_snr":
            return sorted(dets, key=lambda d: d["snr"], reverse=True)
        elif policy == "prefer_short":
            return sorted(dets, key=lambda d: (d["width_samples"], -d["snr"]))
        elif policy == "prefer_long":
            return sorted(dets, key=lambda d: (-d["width_samples"], -d["snr"]))
        else:
            raise ValueError(f"Unknown policy='{policy}'")

    # Process each time-center group independently
    for center_idx, dets in by_time.items():
        if len(dets) == 0:
            continue

        # Initialize occupancy for this time center if needed
        if center_idx not in occ_by_center:
            occ_by_center[center_idx] = np.ones((Ny, Nx), dtype=bool)

        # Sort by policy
        dets_sorted = sort_dets(dets)

        kept = 0
        for d in dets_sorted:
            y = int(d["y"]); x = int(d["x"])
            # If spatial spot already suppressed for this center, skip
            if not occ_by_center[center_idx][y, x]:
                continue

            # Accept detection
            final.append(d)

            # Suppress neighborhood for this center
            y0 = max(0, y - spatial_radius)
            y1 = min(Ny, y + spatial_radius + 1)
            x0 = max(0, x - spatial_radius)
            x1 = min(Nx, x + spatial_radius + 1)
            occ_by_center[center_idx][y0:y1, x0:x1] = False

            # Optional temporal NMS: suppress the same spatial neighborhood in nearby center_idx groups
            if time_radius > 0:
                for dt in range(-time_radius, time_radius + 1):
                    ci2 = center_idx + dt
                    if ci2 == center_idx:
                        continue
                    if ci2 in by_time:
                        if ci2 not in occ_by_center:
                            occ_by_center[ci2] = np.ones((Ny, Nx), dtype=bool)
                        occ_by_center[ci2][y0:y1, x0:x1] = False

            kept += 1
            if (max_per_time_group is not None) and (kept >= max_per_time_group):
                break

    return final

def extract_candidate_snippets(
    times: np.ndarray,
    cube: np.ndarray,                       
    detections: List[Dict[str, Any]],
    *,
    spatial_size: int = 50,                 
    time_factor: int = 5,
    pad_mode: str = "constant",             
    pad_value: float = 0.0,                 
    return_indices: bool = True,            
) -> List[Dict[str, Any]]:

    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1 or time_factor < 1:
        raise ValueError("spatial_size and time_factor must be >= 1")

    half_sp = spatial_size // 2
    out: List[Dict[str, Any]] = []

    # find center time index robustly
    def _resolve_center_idx(det: Dict[str, Any]) -> int:
        if "center_idx" in det:
            return int(det["center_idx"])
        # fallback: nearest time index to time_center
        tc = float(det["time_center"])
        k = np.searchsorted(times, tc)
        if k == 0:
            return 0
        if k >= T:
            return T - 1
        # choose nearest of k-1, k
        return (k - 1) if (abs(times[k-1] - tc) <= abs(times[k] - tc)) else k

    for det in detections:
        y = int(det["y"]); x = int(det["x"])
        w = max(1, int(det["width_samples"]))       # width in samples
        t_center = _resolve_center_idx(det)
        t_len = max(1, time_factor * w)
        half_t = t_len // 2

        # --- Compute desired index ranges (time, y, x) ---
        t0 = t_center - half_t
        t1 = t0 + t_len                             # exclusive

        y0 = y - half_sp
        y1 = y0 + spatial_size                      # exclusive

        x0 = x - half_sp
        x1 = x0 + spatial_size

        # --- Clip to valid indices ---
        t0_clip = max(0, t0)
        t1_clip = min(T, t1)

        y0_clip = max(0, y0)
        y1_clip = min(Ny, y1)

        x0_clip = max(0, x0)
        x1_clip = min(Nx, x1)

        # --- Compute required padding (front/back) to keep fixed size ---
        pad_t_front = t0_clip - t0          # if t0<0 -> positive padding needed
        pad_t_back  = t1 - t1_clip          # if t1>T -> positive padding needed

        pad_y_top   = y0_clip - y0
        pad_y_bot   = y1 - y1_clip

        pad_x_left  = x0_clip - x0
        pad_x_right = x1 - x1_clip

        # --- Slice the valid region ---
        sub_cube  = cube[t0_clip:t1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
        sub_times = times[t0_clip:t1_clip]

        # --- Pad to fixed shapes (t_len, spatial_size, spatial_size) ---
        pad_widths = (
            (pad_t_front, pad_t_back),
            (pad_y_top,   pad_y_bot),
            (pad_x_left,  pad_x_right)
        )

        if pad_mode == "constant":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="constant", constant_values=pad_value)
            # pad times with NaN (no extrapolation)
            snippet_times = np.pad(sub_times, (pad_t_front, pad_t_back), mode="constant", constant_values=np.nan)
        elif pad_mode == "edge":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="edge")
            # for times, duplicate edges
            if sub_times.size == 0:
                # pathological case (w > T): fill with NaN
                snippet_times = np.full((t_len,), np.nan, dtype=times.dtype)
            else:
                front_vals = np.full((pad_t_front,), sub_times[0], dtype=sub_times.dtype)
                back_vals  = np.full((pad_t_back,),  sub_times[-1], dtype=sub_times.dtype)
                snippet_times = np.concatenate([front_vals, sub_times, back_vals], axis=0)
        else:
            raise ValueError(f"Unknown pad_mode='{pad_mode}'")

        # Sanity: enforce exact shapes
        if snippet_cube.shape != (t_len, spatial_size, spatial_size):
            raise RuntimeError(f"Snippet cube has unexpected shape {snippet_cube.shape}")
        if snippet_times.shape != (t_len,):
            raise RuntimeError(f"Snippet times has unexpected shape {snippet_times.shape}")

        # --- Prepare output record ---
        rec: Dict[str, Any] = {
            "candidate": det,
            "snippet_cube": snippet_cube,
            "snippet_times": snippet_times,
        }

        if return_indices:
            rec["meta"] = {
                "time_indices": {"desired": (t0, t1), "clipped": (t0_clip, t1_clip), "pad": (pad_t_front, pad_t_back)},
                "y_indices":    {"desired": (y0, y1), "clipped": (y0_clip, y1_clip), "pad": (pad_y_top, pad_y_bot)},
                "x_indices":    {"desired": (x0, x1), "clipped": (x0_clip, x1_clip), "pad": (pad_x_left, pad_x_right)},
                "center_idx": t_center,
                "snippet_shape": (t_len, spatial_size, spatial_size),
            }

        out.append(rec)

    return out


def augment_candidates_with_world_coords(
    candidates: list[dict],
    *,
    msname: str,
    pixel_size_arcsec: float,
    image_shape: tuple[int, int],
    field_name: str | None = None,
    x0: float | None = None,
    y0: float | None = None,
    x_toward_east: bool = True,
    y_toward_north: bool = True,
) -> dict:
    
    """
    Augment the final candidate list with (l,m) and RA/Dec using TAN projection.

    Parameters
    ----
    
    candidates : list[dict]
        Final filtered candidate dicts (after clustering & dup removal).
        Must include pixel centroid fields: 'x' and 'y' (floats).
    msname : str
        Path to the Measurement Set (to read phase center).
    pixel_size_arcsec : float
        Image pixel size (arcsec / pixel) used to create your final image.
    image_shape : (ny, nx)
         Final image shape to derive default (x0,y0) center if not provided.
    field_name : Optional[str]
         FIELD row name to select phase center (None -> first row)
    x0, y0 : Optional[float]
        Reference pixel coordinates of the phase center.
        If None, defaults to geometric center: ((nx-1)/2, (ny-1)/2).
    x_toward_east, y_toward_north : bool
        Axis orientation flags.
    
    Returns
    -------
    dict
        {
          "candidates": <list[dict] with added world keys>,
          "world": <World>,
        }
    """
    
    ny, nx = image_shape
    if x0 is None:
        x0 = (nx - 1) / 2.0
    if y0 is None:
        y0 = (ny - 1) / 2.0

    ra0_rad, dec0_rad, picked_field = get_phase_center_from_ms(msname, field_name)
    world = World(
        ra0_rad=ra0_rad,
        dec0_rad=dec0_rad,
        pixel_scale_arcsec=pixel_size_arcsec,
        x0=float(x0),
        y0=float(y0),
        field_name=picked_field,
        x_toward_east=x_toward_east,
        y_toward_north=y_toward_north,
    )

    # Vectorize inputs for speed
    xs = np.array([c["x"] for c in candidates], dtype=np.float64)
    ys = np.array([c["y"] for c in candidates], dtype=np.float64)

    l, m = lm_from_pixel_xy(xs, ys, world)
    ra_rad, dec_rad = radec_from_lm(l, m, world)

    # Attach results to candidate dicts
    for i, c in enumerate(candidates):
        c.update({
            # l,m in radians and arcsec
            "l_rad": float(l[i]),
            "m_rad": float(m[i]),
            "l_arcsec": float(l[i] / ARCSEC_TO_RAD),
            "m_arcsec": float(m[i] / ARCSEC_TO_RAD),
            # RA/Dec
            "ra_rad": float(ra_rad[i]),
            "dec_rad": float(dec_rad[i]),
            "ra_deg": float(ra_rad[i] * RAD_TO_DEG),
            "dec_deg": float(dec_rad[i] * RAD_TO_DEG),
            # phase center context
            "phase_center_ra_rad": world.ra0_rad,
            "phase_center_dec_rad": world.dec0_rad,
            "field_name": world.field_name,
        })

    return {"candidates": candidates, "world": world}



def main():
    parser = argparse.ArgumentParser(description='Image MS in time chunks using ducc0 wgridder')
    parser.add_argument('--msname', required=True, help='Path to Measurement Set')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of time samples per chunk (default: 1000)')
    parser.add_argument('--corr-mode', choices=['average','stokesI','single'], default='single',
                        help='Correlation handling mode (default: single)')
    parser.add_argument('--basis', choices=['auto','linear','circular'], default='linear',
                        help='Basis for stokesI (default: auto)')
    parser.add_argument('--single-pol', default='XX',
                        help='Single pol to image when corr-mode=single (default: XX)')
    parser.add_argument('--data-column', default='DATA', help='Which data column to image from the measurement set (default: DATA)')
    parser.add_argument('--npix-x', type=int, default=384)
    parser.add_argument('--npix-y', type=int, default=384)
    parser.add_argument('--pixsize-arcsec', type=float, default=22.0,
                        help='Pixel size (arcsec); applied to both axes (default: 22.0)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--do-wgridding', dest='do_wgridding', action='store_true', default=True)
    parser.add_argument('--no-wgridding', dest='do_wgridding', action='store_false')
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--do-plot', action='store_true')
    args = parser.parse_args()

    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0

    # Discover total number of time chunks via iter
    t_main = table(args.msname, readonly=True)
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    #it.close()
    t_main.close()

    print(f"Found {total_chunks} time chunks in MS: {args.msname}")

    start = 0
    chunk_size = max(1, args.chunk_size)
    chunk_id = 0

    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        print(f"[Chunk {chunk_id}] imaging time_idx {start}..{end}")
        times, cube = image_time_samples(msname=args.msname,
                                         start_time_idx=start,
                                         end_time_idx=end,
                                         corr_mode=args.corr_mode,
                                         basis=args.basis,
                                         single_pol=args.single_pol,
                                         data_column=args.data_column,
                                         npix_x=args.npix_x,
                                         npix_y=args.npix_y,
                                         pixsize_x=pix_rad,
                                         pixsize_y=pix_rad,
                                         epsilon=args.epsilon,
                                         do_wgridding=args.do_wgridding,
                                         nthreads=args.nthreads,
                                         verbosity=args.verbosity,
                                         do_plot=args.do_plot,
                                         )

        print(len(times), cube.shape)
        detections, snr_cubes = boxcar_search_time(times, cube,
                                                   widths=[1, 2, 4, 8, 16, 32, 64],
                                                   widths_in_seconds=False,      # set True if widths are in seconds
                                                   threshold_sigma=7.5,
                                                   return_snr_cubes=True,
                                                   keep_top_k=50,                # keep top 50 per width
                                                   std_mode="spatial_per_window",
                                                   subtract_mean_per_pixel=True  # high-pass in time per pixel
                                                   )

        # 1) NMS per width (spatial, optional temporal)
        detections_by_width = nms_snr_maps_per_width(
            snr_cubes, times,
            threshold_sigma=7.5,
            spatial_radius=2,
            time_radius=0,          # set >0 to suppress across neighboring time slices too
            valid_mask=None
        )
        
        # 2) Merge across widths and apply cross-width NMS
        final_detections = group_filter_across_widths(
            detections_by_width, times,
            spatial_radius=2,
            time_radius=0,          # set >0 to suppress across nearby center_idx groups
            policy="max_snr",       # or "prefer_short"/"prefer_long"
            max_per_time_group=1,   # keep only 1 per time center (typical)
            ny_nx=(cube.shape[1], cube.shape[2])  # pass explicit Ny,Nx for speed
        )


        
        print(f"chunk start:{start} end:{end}: Found {len(final_detections)} candidates")

        if len(final_detections) > 0:
            snippets = extract_candidate_snippets(
                times, cube, final_detections,
                spatial_size=50,
                time_factor=5,
                pad_mode="constant",   # or "edge"
                pad_value=0.0,
                return_indices=True
            )
            #import ipdb; ipdb.set_trace()
            print(f"Prepared {len(snippets)} snippets.")
            # For the first snippet:
            first = snippets[0]
            print("Candidate:", first["candidate"])
            print("Snippet cube shape:", first["snippet_cube"].shape)      # (5*w, 50, 50)
            
            plt.imshow(first["snippet_cube"][first["snippet_cube"].shape[0]//2 +1, :, :], aspect='auto', origin='lower')
            plt.savefig("test_snippet.png", dpi=300)
            print("Snippet times shape:", first["snippet_times"].shape)    # (5*w,)
            print("Meta:", first["meta"])

        
        
        start = end + 1
        chunk_id += 1
        
if __name__ == '__main__':
    main()
