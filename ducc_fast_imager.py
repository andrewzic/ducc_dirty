import argparse
import glob
import os
import shutil
import sys
from tqdm import tqdm
import numpy as np
import astropy.constants as const
import astropy.units as u
from casacore.tables import table
from typing import Iterable, Tuple, List, Dict, Any, Optional


# CASA POLARIZATION CORR_TYPE integer codes to labels
_CORR_CODE_TO_NAME = {
    5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL',
    9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY',
}

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
        
        if t_chunk_idx == 10:
            print("adding fake transient")
            dirty[198:202,198:202] = 100.0
            
        cube[cube_idx, :, :] = dirty
        cube_idx += 1

        if do_plot:
            import matplotlib.pyplot as plt
            plt.imshow(dirty, aspect='auto', interpolation='none', origin='lower', cmap='gray')
            plt.savefig(f"img_{cube_idx:04d}.png", dpi=300)
            sys.exit()
            plt.show()
        #results.append((time_val, dirty))
        
    #it.close()
    t_main.close()
    
    results = (times, cube)    
    return results


def boxcar_search_time(
    times: np.ndarray,
    cube: np.ndarray,                      # shape (T, Ny, Nx)
    widths: Iterable[float],               # list of widths; samples (int) or seconds (float)
    *,
    widths_in_seconds: bool = False,
    threshold_sigma: float = 5.0,
    return_snr_cubes: bool = False,
    keep_top_k: Optional[int] = None,      # per width, cap detections to top-K by SNR
    valid_mask: Optional[np.ndarray] = None,  # (Ny, Nx) mask of pixels to search
    subtract_mean_per_pixel: bool = False,    # high-pass in time per pixel
    std_mode: str = "spatial_per_window",     # "spatial_per_window" | "temporal_per_pixel"
    spatial_estimator: str = "clipped_rms",           # "mad" | "clipped_rms"
    clip_sigma: float = 3.0,                  # used when spatial_estimator="clipped_rms"
) -> Tuple[List[Dict[str, Any]], Optional[Dict[int, np.ndarray]]]:
    """
    Boxcar search along time axis with choice of std estimator.

    std_mode == "spatial_per_window":
      For each width w and window index t, compute S_w[t,:,:] by summing w frames.
      Estimate a single spatial std sigma_w[t] from S_w[t,:,:] and define
      SNR_w[t,y,x] = S_w[t,y,x] / sigma_w[t].

    std_mode == "temporal_per_pixel":
      (Original) Per-pixel temporal variance via cumulative sums of x and x^2.

    spatial_estimator:
      - "mad": robust std via 1.4826 * MAD over (y,x)
      - "clipped_rms": iteratively sigma-clipped RMS over (y,x)

    Other args as before.
    """
    # Basic shape checks
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")

    # Optional pixel mask
    if valid_mask is None:
        valid_mask = np.ones((Ny, Nx), dtype=bool)
    elif valid_mask.shape != (Ny, Nx):
        raise ValueError("valid_mask must be shape (Ny, Nx)")

    # Optional temporal high-pass
    data = cube.astype(np.float64, copy=False)  # use float64 for stable stats
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
            w_samp = max(1, int(round(w_sec / dt)))
            widths_samples.append(w_samp)
    else:
        for w in widths:
            widths_samples.append(max(1, int(w)))

    detections: List[Dict[str, Any]] = []
    snr_cubes: Optional[Dict[int, np.ndarray]] = {} if return_snr_cubes else None

    # Precompute cumulative sums along time for fast windowed sums
    csum  = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum[1:] = np.cumsum(data, axis=0)

    def moving_sum(w: int) -> np.ndarray:
        # sum over [t0, t0+w) for t0 = 0..T-w
        return csum[w:] - csum[:-w]  # (T-w+1, Ny, Nx)

    # Helper: robust spatial std estimators on S_w[t,:,:]
    def spatial_std_mad(Sw: np.ndarray, mask3: np.ndarray) -> np.ndarray:
        # Sw: (T_eff, Ny, Nx), mask3: (1, Ny, Nx) broadcastable
        # masked median
        S_masked = np.where(mask3, Sw, np.nan)
        med = np.nanmedian(S_masked, axis=(1, 2))  # (T_eff,)
        mad = np.nanmedian(np.abs(S_masked - med[:, None, None]), axis=(1, 2))  # (T_eff,)
        return 1.4826 * mad  # robust std

    def spatial_std_clipped_rms(Sw: np.ndarray, mask3: np.ndarray, sigma: float, max_iter: int = 5) -> np.ndarray:
        # Iterative sigma clipping on (y,x) for each time window
        T_eff = Sw.shape[0]
        sigmas = np.empty((T_eff,), dtype=np.float64)
        for t0 in range(T_eff):
            arr = Sw[t0]
            arr = np.where(mask3[0], arr, np.nan)
            # initial guess: MAD
            med0 = np.nanmedian(arr)
            mad0 = np.nanmedian(np.abs(arr - med0))
            std = 1.4826 * mad0 if np.isfinite(mad0) else np.nanstd(arr)
            mu = med0
            for _ in range(max_iter):
                if not np.isfinite(std) or std <= 0:
                    break
                keep = np.abs(arr - mu) <= sigma * std
                if not np.any(keep & ~np.isnan(arr)):
                    break
                mu  = np.nanmean(arr[keep])
                std = np.nanstd(arr[keep], ddof=1)
            sigmas[t0] = std if np.isfinite(std) and std > 0 else np.nanstd(arr, ddof=1)
        # Replace non-finite by small epsilon to avoid div-by-zero
        sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0), sigmas, np.nanmedian(sigmas[np.isfinite(sigmas) & (sigmas > 0)]) or 1.0)
        return sigmas

    mask3 = valid_mask[None, :, :]  # (1, Ny, Nx) for broadcasting

    for w in widths_samples:
        if w > T:
            continue

        # Windowed sums (this already includes sqrt(w) scale if noise is white)
        S = moving_sum(w)  # (T_eff, Ny, Nx)
        T_eff = S.shape[0]

        if std_mode == "spatial_per_window":
            if spatial_estimator == "mad":
                sigma_w = spatial_std_mad(S, mask3)             # (T_eff,)
            elif spatial_estimator == "clipped_rms":
                sigma_w = spatial_std_clipped_rms(S, mask3, clip_sigma)
            else:
                raise ValueError(f"Unknown spatial_estimator='{spatial_estimator}'")
            # SNR uses per-window spatial std
            snr = S / sigma_w[:, None, None]                    # (T_eff, Ny, Nx)

        elif std_mode == "temporal_per_pixel":
            # Original method: per-pixel temporal variance in the window
            # (compute s and std via csum/csum2 like you had)
            csum2 = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
            csum2[1:] = np.cumsum(data * data, axis=0)
            s  = csum[w:]  - csum[:-w]
            s2 = csum2[w:] - csum2[:-w]
            mean = s / w
            var  = s2 / w - mean * mean
            var  = np.clip(var, a_min=0.0, a_max=None)
            std = np.sqrt(var)
            denom = std * np.sqrt(w)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.where(denom > 0.0, s / denom, 0.0)
        else:
            raise ValueError(f"Unknown std_mode='{std_mode}'")

        # Respect valid_mask (optional extra gating)
        if valid_mask is not None:
            snr = np.where(mask3, snr, 0.0)

        # Optionally return SNR cube
        if return_snr_cubes:
            snr_cubes[w] = snr.astype(np.float32, copy=False)  # (T_eff, Ny, Nx)

        # Threshold and gather detections
        hits = np.where(snr >= threshold_sigma)
        t0_idx, ys, xs = hits
        if t0_idx.size == 0:
            continue

        # Time-center for window [t0, t0+w)
        centers = times[t0_idx + (w // 2)]

        # Top-K per width
        if keep_top_k is not None and t0_idx.size > keep_top_k:
            order = np.argsort(snr[t0_idx, ys, xs])[::-1]
            sel   = order[:keep_top_k]
            t0_idx, ys, xs, centers = t0_idx[sel], ys[sel], xs[sel], centers[sel]

        # Append

        for i in range(t0_idx.size):
            det = {
                "time_center": float(centers[i]),
                "y": int(ys[i]),
                "x": int(xs[i]),
                "width_samples": int(w),
                "snr": float(snr[t0_idx[i], ys[i], xs[i]]),
                "value_sum": float(S[t0_idx[i], ys[i], xs[i]]) if std_mode == "spatial_per_window" else float(s[t0_idx[i], ys[i], xs[i]]),
                # NEW: indices (helps snippet extraction)
                "t0_idx": int(t0_idx[i]),                       # window start index
                "center_idx": int(t0_idx[i] + (w // 2)),        # center index in time axis
            }
            detections.append(det)
            
    return detections, snr_cubes



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

    # Helper: find center time index robustly
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
                                                   widths=[1], #, 2, 4, 8, 16, 32, 64, 128],
                                                   widths_in_seconds=False,      # set True if widths are in seconds
                                                   threshold_sigma=7.5,
                                                   return_snr_cubes=True,
                                                   keep_top_k=50,                # keep top 50 per width
                                                   std_mode="spatial_per_window",
                                                   subtract_mean_per_pixel=True  # high-pass in time per pixel
                                                   )


        
        print(f"chunk start:{start} end:{end}: Found {len(detections)} candidates")

        
        snippets = extract_candidate_snippets(
            times, cube, detections,
            spatial_size=50,
            time_factor=5,
            pad_mode="constant",   # or "edge"
            pad_value=0.0,
            return_indices=True
        )

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
