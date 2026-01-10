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
from ligo.skymap.plot.marker import reticle

from scipy.ndimage import maximum_filter

from casacore.tables import table
try:
    import ducc0
except Exception as e:
    raise RuntimeError('ducc0 is required') from e



from fastducc import wcs as ducc_wcs
from fastducc import filters, kernels, ms_utils, detection

def make_stdmap_snippet(
    std_map: np.ndarray,                # (Ny, Nx)
    candidate: Dict[str, Any],
    spatial_size: int = 50
) -> Dict[str, Any]:
    """
    Create a snippet record from a 2-D variance/std map.
    Time length is 1 (single frame), with NaN time stamp for compatibility.
    """
    Ny, Nx = std_map.shape
    y = int(candidate["y"]); x = int(candidate["x"])
    half_sp = spatial_size // 2

    y0 = max(0, y - half_sp)
    y1 = min(Ny, y0 + spatial_size)
    x0 = max(0, x - half_sp)
    x1 = min(Nx, x0 + spatial_size)

    cut = std_map[y0:y1, x0:x1]
    # shape should be exactly (spatial_size, spatial_size); pad if cropped at edges
    pad_y_top = max(0, (y - half_sp) - y0)
    pad_y_bot = max(0, (y0 + spatial_size) - y1)
    pad_x_left = max(0, (x - half_sp) - x0)
    pad_x_right = max(0, (x0 + spatial_size) - x1)

    snippet_2d = np.pad(cut,
                        ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right)),
                        mode="edge")
    # Time length = 1
    snippet_cube = snippet_2d[None, :, :]  # (1, spatial_size, spatial_size)
    snippet_times = np.array([np.nan], dtype=float)

    rec = {
        "candidate": candidate,
        "snippet_cube": snippet_cube.astype(np.float32),
        "snippet_times": snippet_times,
        "meta": {
            "time_indices": {"desired": (0, 1), "clipped": (0, 1), "pad": (0, 0)},
            "y_indices":    {"desired": (y - half_sp, y - half_sp + spatial_size),
                             "clipped": (y0, y1), "pad": (pad_y_top, pad_y_bot)},
            "x_indices":    {"desired": (x - half_sp, x - half_sp + spatial_size),
                             "clipped": (x0, x1), "pad": (pad_x_left, pad_x_right)},
            "center_idx": 0,
            "snippet_shape": (1, spatial_size, spatial_size),
        }
    }
    return rec


def extract_candidate_snippets(
    times: np.ndarray,
    cube: np.ndarray,
    detections: List[Dict[str, Any]],
    *,
    spatial_size: int = 50,
    time_factor: int = 5,            # in units of *smoothed* frames
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    return_indices: bool = True,
    center_policy: str = "right",    # 'right' uses w//2, 'left' uses (w-1)//2 for center time
) -> List[Dict[str, Any]]:
    """
    Extract candidate snippets from a *boxcar-smoothed* version of the cube,
    using the candidate width as the temporal averaging window.

    Parameters
    ----------
    times : (T,) ndarray
        Original time stamps (seconds or any monotonically increasing floats).
    cube : (T, Ny, Nx) ndarray
        Original full-resolution image cube.
    detections : list of dict
        Filtered candidates (already annotated with sky coordinates), each with at least:
           'y','x' : int pixel coordinates in the full image
           'width_samples' : int boxcar width (samples)
           't0_idx' : int window start index (recommended)
           or 'center_idx' : int original center sample (falls back if t0_idx missing)
    spatial_size : int
        Spatial crop size (square cutout).
    time_factor : int
        Snippet temporal length measured in *smoothed* frames (not original frames).
        Final snippet length is `t_len_sm = max(1, time_factor)`.
    pad_mode : {"constant","edge"}
        Padding mode to enforce fixed snippet shape.
    pad_value : float
        Constant padding value if pad_mode="constant".
    return_indices : bool
        If True, include index bookkeeping in the "meta" dictionary.
    center_policy : {"right","left"}
        Policy for defining the *center time* of each w-sample window:
          - "right": center = t0 + w//2 (default; consistent with previous fixes)
          - "left":  center = t0 + (w-1)//2

    Returns
    -------
    out : list of dict
        Each record:
          {
            "candidate": det_with_extra_fields,
            "snippet_cube": (t_len_sm, spatial_size, spatial_size) array,
            "snippet_times": (t_len_sm,) array,
            "meta": {...}  # if return_indices=True
          }
    """
    # --- validate inputs ---
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1 or time_factor < 1:
        raise ValueError("spatial_size and time_factor must be >= 1")

    # --- spatial center index within snippet ---
    half_sp = spatial_size // 2

    # --- Precompute cumulative sum for fast boxcar smoothing (once) ---
    # Use float64 for numerical stability (same dtype as numba kernels).
    csum = np.zeros((T + 1, Ny, Nx), dtype=np.float64)
    csum[1:] = np.cumsum(cube.astype(np.float64, copy=False), axis=0)


    # --- Build smoothed cubes & times per unique width ---
    unique_w = sorted(set(max(1, int(d.get("width_samples", 1))) for d in detections))
    smoothed_by_w: Dict[int, Dict[str, Any]] = {}

    for w in unique_w:
        if w > T:
            # pathological: window longer than data
            sm_cube = np.empty((0, Ny, Nx), dtype=np.float64)
            sm_times = np.empty((0,), dtype=times.dtype)
        else:
            S_sum = kernels._moving_sum_from_csum(csum, w)  # (T_eff, Ny, Nx) windowed sums
            sm_cube = S_sum / float(w)
            T_eff = sm_cube.shape[0]
            # Choose center time index inside each window
            offset = (w // 2) if (center_policy == "right") else ((w - 1) // 2)
            # `times_sm[t0] = times[t0 + offset]`
            start = offset
            end = start + T_eff
            sm_times = times[start:end]
        smoothed_by_w[w] = {"cube": sm_cube, "times": sm_times}

    out: List[Dict[str, Any]] = []

    # --- helper to resolve indices per detection ---
    def _resolve_w(det: Dict[str, Any]) -> int:
        return max(1, int(det.get("width_samples", 1)))

    def _resolve_sm_center_idx(det: Dict[str, Any], w: int) -> int:
        """
        Center index in *smoothed* series. Prefer t0_idx if present; otherwise
        map from original center_idx back to window start.
        """
        if "t0_idx" in det:
            return int(det["t0_idx"])
        # Fallback: derive t0_idx from center_idx and width
        c = int(det.get("center_idx", 0))
        # For 'right' center, center_idx = t0 + w//2  -> t0 = center_idx - w//2
        # For 'left'  center, center_idx = t0 + (w-1)//2
        if center_policy == "right":
            return max(0, c - (w // 2))
        else:
            return max(0, c - ((w - 1) // 2))

    # --- main loop over detections ---
    for det in detections:
        y = int(det["y"]); x = int(det["x"])
        w = _resolve_w(det)

        sm = smoothed_by_w[w]
        sm_cube = sm["cube"]         # shape: (T_eff, Ny, Nx)
        sm_times = sm["times"]       # shape: (T_eff,)
        T_eff = sm_cube.shape[0]

        # If width exceeds data length, return an all-NaN snippet
        if T_eff == 0:
            t_len_sm = max(1, time_factor)
            snippet_cube = np.full((t_len_sm, spatial_size, spatial_size), np.nan, dtype=np.float32)
            snippet_times = np.full((t_len_sm,), np.nan, dtype=times.dtype)
            rec = {"candidate": det, "snippet_cube": snippet_cube, "snippet_times": snippet_times}
            if return_indices:
                rec["meta"] = {
                    "time_indices": {"desired": (0, t_len_sm), "clipped": (0, 0), "pad": (t_len_sm, 0)},
                    "y_indices": {"desired": (y - half_sp, y - half_sp + spatial_size),
                                  "clipped": (0, 0), "pad": (spatial_size, 0)},
                    "x_indices": {"desired": (x - half_sp, x - half_sp + spatial_size),
                                  "clipped": (0, 0), "pad": (spatial_size, 0)},
                    "center_idx_smoothed": 0,
                    "snippet_shape": (t_len_sm, spatial_size, spatial_size),
                }
            out.append(rec)
            continue

        # Smoothed center index (frame in smoothed series)
        k_center = _resolve_sm_center_idx(det, w)

        # Snippet length in *smoothed frames*
        t_len_sm = max(1, time_factor)
        half_t = t_len_sm // 2

        # Desired smoothed time window around k_center
        k0 = k_center - half_t
        k1 = k0 + t_len_sm  # exclusive

        # Clip to valid smoothed indices
        k0_clip = max(0, k0)
        k1_clip = min(T_eff, k1)

        # Spatial desired indices (same as before, about full-resolution pixel location)
        y0 = y - half_sp
        y1 = y0 + spatial_size
        x0 = x - half_sp
        x1 = x0 + spatial_size

        # Clip in space
        y0_clip = max(0, y0)
        y1_clip = min(Ny, y1)
        x0_clip = max(0, x0)
        x1_clip = min(Nx, x1)

        # Required padding to keep fixed shapes
        pad_t_front = k0_clip - k0
        pad_t_back  = k1 - k1_clip
        pad_y_top   = y0_clip - y0
        pad_y_bot   = y1 - y1_clip
        pad_x_left  = x0_clip - x0
        pad_x_right = x1 - x1_clip

        # Slice smoothed region
        sub_cube  = sm_cube[k0_clip:k1_clip, y0_clip:y1_clip, x0_clip:x1_clip]
        sub_times = sm_times[k0_clip:k1_clip]

        # Pad to fixed shapes
        pad_widths = (
            (pad_t_front, pad_t_back),
            (pad_y_top,   pad_y_bot),
            (pad_x_left,  pad_x_right),
        )

        if pad_mode == "constant":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="constant", constant_values=pad_value)
            snippet_times = np.pad(sub_times, (pad_t_front, pad_t_back),
                                   mode="constant", constant_values=np.nan)
        elif pad_mode == "edge":
            snippet_cube = np.pad(sub_cube, pad_widths, mode="edge")
            if sub_times.size == 0:
                snippet_times = np.full((t_len_sm,), np.nan, dtype=times.dtype)
            else:
                front_vals = np.full((pad_t_front,), sub_times[0], dtype=sub_times.dtype)
                back_vals  = np.full((pad_t_back,),  sub_times[-1], dtype=sub_times.dtype)
                snippet_times = np.concatenate([front_vals, sub_times, back_vals], axis=0)
        else:
            raise ValueError(f"Unknown pad_mode='{pad_mode}'")

        # Enforce exact shapes
        if snippet_cube.shape != (t_len_sm, spatial_size, spatial_size):
            raise RuntimeError(f"Snippet cube has unexpected shape {snippet_cube.shape}")
        if snippet_times.shape != (t_len_sm,):
            raise RuntimeError(f"Snippet times has unexpected shape {snippet_times.shape}")

        # Optional: local peak search around the smoothed center frame
        lc = snippet_cube[:, half_sp, half_sp]  # detection pixel (center)
        good = np.isfinite(snippet_times) & np.isfinite(lc)
        if np.any(good):
            rel_peak = np.nanargmax(lc[good])
            # Map back to snippet indices: among good only; use direct max over snippet for simplicity
            rel_peak_all = int(np.nanargmax(lc))
            det_time_peak = snippet_times[rel_peak_all]
            det["time_center_peak_smoothed"] = float(det_time_peak) if np.isfinite(det_time_peak) else float(det.get("time_center", np.nan))
            det["center_idx_peak_smoothed"]  = int(k_center + (rel_peak_all - half_t))

        # Record smoothed center time (for plotting)
        det["time_center_smoothed"] = float(sm_times[k_center]) if (0 <= k_center < sm_times.shape[0]) else float(det.get("time_center", np.nan))
        det["center_idx_smoothed"]  = int(k_center)

        # Prepare output record
        rec: Dict[str, Any] = {
            "candidate": det,
            "snippet_cube": snippet_cube.astype(np.float32, copy=False),
            "snippet_times": snippet_times,
        }

        if return_indices:
            rec["meta"] = {
                "time_indices": {"desired": (k0, k1), "clipped": (k0_clip, k1_clip), "pad": (pad_t_front, pad_t_back)},
                "y_indices":    {"desired": (y0, y1), "clipped": (y0_clip, y1_clip), "pad": (pad_y_top, pad_y_bot)},
                "x_indices":    {"desired": (x0, x1), "clipped": (x0_clip, x1_clip), "pad": (pad_x_left, pad_x_right)},
                "center_idx_smoothed": int(k_center),
                "snippet_shape": (t_len_sm, spatial_size, spatial_size),
                "width_samples": int(w),
            }

        out.append(rec)

    return out

def candidates_to_astropy_table(annotated: List[Dict[str, Any]]) -> Table:
    """
    Convert the list of annotated candidate dicts into an Astropy Table with units.

    Expected keys per candidate dict (from annotate_candidates_with_sky_coords):
      'x','y','l','m','ra_rad','dec_rad','ra_hms','dec_dms','snr','width_samples',
      'time_start','time_end','time_center','t0_idx','t1_idx_excl','center_idx',
      'phase_center_field' (some keys may be missing depending on pipeline)

    Returns
    -------
    t : astropy.table.Table
        Table with appropriate columns and units.
    """
    if len(annotated) == 0:
        # return an empty table with typical schema
        t = Table(names=[
            "x","y","l","m","ra_rad","dec_rad","ra_deg","dec_deg",
            "ra_hms","dec_dms","snr","width_samples",
            "time_start","time_end","time_center",
            "t0_idx","t1_idx_excl","center_idx","phase_center_field"
        ], dtype=[
            int,int,float,float,float,float,float,float,
            "U20","U20",float,int,
            float,float,float,
            int,int,int,"U64"
        ])
        return t

    # Build arrays column-wise, handling missing keys gracefully
    def col(key, default=np.nan):
        return [c.get(key, default) for c in annotated]

    # Base numeric columns
    xs  = np.array(col("x", default=np.int64(-1)), dtype=np.int64)
    ys  = np.array(col("y", default=np.int64(-1)), dtype=np.int64)
    ls  = np.array(col("l"), dtype=float)
    ms  = np.array(col("m"), dtype=float)
    ra_rad = np.array(col("ra_rad"), dtype=float)
    dec_rad = np.array(col("dec_rad"), dtype=float)

    # Derived in degrees for convenience
    ra_deg  = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    # Strings for sexagesimal
    ra_hms  = np.array(col("ra_hms", default=""), dtype=str)
    dec_dms = np.array(col("dec_dms", default=""), dtype=str)

    # Optional metrics
    snr          = np.array(col("snr"), dtype=float)
    width_samples= np.array(col("width_samples", default=np.int64(-1)), dtype=np.int64)

    # Times & indices (float seconds, int indices)
    time_start   = np.array(col("time_start"), dtype=float)
    time_end     = np.array(col("time_end"), dtype=float)
    time_center  = np.array(col("time_center"), dtype=float)

    t0_idx       = np.array(col("t0_idx", default=np.int64(-1)), dtype=np.int64)
    t1_idx_excl  = np.array(col("t1_idx_excl", default=np.int64(-1)), dtype=np.int64)
    center_idx   = np.array(col("center_idx", default=np.int64(-1)), dtype=np.int64)

    # Field name (string)
    field_name   = np.array(col("phase_center_field", default=""), dtype=str)

    # Build Astropy Table
    t = Table()
    t["x"] = xs
    t["y"] = ys
    t["l"] = ls * u.dimensionless_unscaled          # direction cosine
    t["m"] = ms * u.dimensionless_unscaled          # direction cosine
    t["ra_rad"]  = ra_rad * u.rad
    t["dec_rad"] = dec_rad * u.rad
    t["ra_deg"]  = ra_deg * u.deg
    t["dec_deg"] = dec_deg * u.deg
    t["ra_hms"]  = ra_hms
    t["dec_dms"] = dec_dms
    t["snr"]     = snr
    t["width_samples"] = width_samples
    t["time_start"]  = time_start * u.s
    t["time_end"]    = time_end * u.s
    t["time_center"] = time_center * u.s
    t["t0_idx"]      = t0_idx
    t["t1_idx_excl"] = t1_idx_excl
    t["center_idx"]  = center_idx
    t["phase_center_field"] = field_name

    return t

def save_candidates_table(table: Table, csv_path: str, vot_path: str) -> None:
    """
    Save the Astropy Table to CSV and VOTable.
    - CSV: plain text; units appear in metadata but not in numeric values.
    - VOTable: includes unit information in FIELDs where available.

    Parameters
    ----------
    table : astropy.table.Table
        The table to write.
    csv_path : str
        Destination CSV file path.
    vot_path : str
        Destination VOTable file path (usually .vot or .xml).
    """
    # CSV
    table.write(csv_path, format="csv", overwrite=True)

    # VOTable (Astropy handles units and metadata)
    table.write(vot_path, format="votable", overwrite=True)


def save_candidate_lightcurves(
    times: np.ndarray,
    cube: np.ndarray,
    candidate: Dict[str, Any],
    out_prefix: str,
    std_map: Optional[np.ndarray] = None,
    use_std_images: bool = False,
    *,
    spatial_size: int = 50,
    save_format: str = "npz",        # "npz" or "ascii"
    center_policy: str = "right",    # "right" uses w//2; "left" uses (w-1)//2
    cmap: str = "gray",
    dpi: int = 200,
    # WCS parameters:
    npix_x: int = None, npix_y: int = None,  # full image size (defaults from cube if None)
    ra0_rad: float = None, dec0_rad: float = None,  # phase center (radians)
    pix_rad: float = None,                  # pixel scale (radians/pixel) from CLI
    ra_sign: int = -1,                      # -1 => RA increases left; +1 => RA increases right
    dec_sign: int = 1,
    radesys: str = "ICRS", equinox: float = None
) -> Dict[str, str]:


    """
    Extract and save full-resolution and boxcar-smoothed light curves for a candidate,
    and produce a WCS-labelled figure:
      - Top-left: full frame at detection time (WCS RA/Dec)
      - Top-right: cutout/snippet around candidate (WCS RA/Dec)
      - Middle: full-res LC
      - Bottom: boxcar-smoothed LC

    Called before snippet products are produced.

    Returns dict of file paths.
    """

    
    # --- Validate inputs ---
    if cube.ndim != 3:
        raise ValueError("cube must be (T, Ny, Nx)")
    T, Ny, Nx = cube.shape
    if times.shape[0] != T:
        raise ValueError("times length must match cube time axis")
    if spatial_size < 1:
        raise ValueError("spatial_size must be >= 1")
    if save_format not in ("npz", "ascii"):
        raise ValueError("save_format must be 'npz' or 'ascii'")
    if pix_rad is None or ra0_rad is None or dec0_rad is None:
        raise ValueError("You must pass ra0_rad/dec0_rad (phase center) and pix_rad (CLI pixel scale).")

    # Default full image dims from cube if not provided
    if npix_x is None: npix_x = Nx
    if npix_y is None: npix_y = Ny

    # --- Candidate fields ---
    y = int(candidate["y"])
    x = int(candidate["x"])
    w = max(1, int(candidate.get("width_samples", 1)))

    # Preferred smoothed start index, else derive from center_idx
    if "t0_idx" in candidate:
        
        t0_idx = int(candidate["t0_idx"])
        
    else:
        cidx = int(candidate.get("center_idx", 0))
        t0_idx = cidx - (w // 2) if (center_policy == "right") else cidx - ((w - 1) // 2)
        t0_idx = max(0, min(t0_idx, T - w))

    offset = (w // 2) if (center_policy == "right") else ((w - 1) // 2)
    t_full_center = max(0, min(t0_idx + offset, T - 1))

    # --- Full-res light curve at the candidate pixel ---
    lc_full = cube[:, y, x].astype(np.float64, copy=False)

    # --- Boxcar-smoothed light curve ---
    lc_sm, T_eff = kernels._compute_boxcar_1d(lc_full, w)
    if T_eff > 0:
        times_sm = times[offset:offset + T_eff]
        k_center = max(0, min(t0_idx, T_eff - 1))  # smoothed index of detection window start
    else:
        times_sm = np.empty((0,), dtype=times.dtype)
        k_center = 0

    # --- Images to display ---

    if use_std_images and (std_map is not None):
        # Use std_map for both full frame and cutout
        frame_for_display = std_map
    else:
        frame_for_display = cube[t_full_center]
        
    vmin = np.nanpercentile(frame_for_display, 5.0)
    vmax = np.nanpercentile(frame_for_display, 99.5)
    
    # Full-frame WCS:
    wcs_full = ducc_wcs._build_fullframe_wcs(npix_x=npix_x, npix_y=npix_y,
                                    ra0_rad=ra0_rad, dec0_rad=dec0_rad,
                                    pixscale_rad=pix_rad,
                                    ra_sign=ra_sign, radesys=radesys, equinox=equinox)


    if "ra_rad" in candidate and "dec_rad" in candidate:
        ra_c = float(candidate["ra_rad"])
        dec_c = float(candidate["dec_rad"])
    else:

        ra_c = ra0_rad
        dec_c = dec0_rad
    
    # Snippet/cutout for the chosen frame
    half_sp = spatial_size // 2
    y0 = max(0, y - half_sp); y1 = min(Ny, y0 + spatial_size)
    x0 = max(0, x - half_sp); x1 = min(Nx, x0 + spatial_size)
    snippet = frame_for_display[y0:y1, x0:x1]
    
    # Snippet WCS centered on candidate world coords
    wcs_cut = ducc_wcs._build_tan_wcs_for_snippet(spatial_size=snippet.shape[0],
                                         ra_rad=ra_c, dec_rad=dec_c,
                                         pixscale_rad=pix_rad,
                                         ra_sign=ra_sign, radesys=radesys, equinox=equinox)



    # --- Save light curves ---
    out_full = f"{out_prefix}_lc_full.{ 'npz' if save_format=='npz' else 'txt' }"
    out_box  = f"{out_prefix}_lc_boxcar_w{w}.{ 'npz' if save_format=='npz' else 'txt' }"

    if save_format == "npz":
        np.savez(out_full, time=times, flux=lc_full)
        np.savez(out_box, time=times_sm, flux=lc_sm, width=w)
    else:
        # ASCII: two columns "time flux" with header
        with open(out_full, "w") as f:
            f.write("# time  flux(full_res)\n")
            for t, v in zip(times, lc_full):
                f.write(f"{t:.9f} {v:.9e}\n")
        with open(out_box, "w") as f:
            f.write(f"# time  flux(boxcar_mean_w={w})\n")
            for t, v in zip(times_sm, lc_sm):
                f.write(f"{t:.9f} {v:.9e}\n")

    
    
    # --- Build the figure ---
    out_fig = f"{out_prefix}_lightcurves.png"
    fig = plt.figure(figsize=(10, 8), dpi=dpi)

    
    # Top-left: WCS full frame
    ax_full = fig.add_subplot(2, 2, 1, projection=wcs_full)
    im_full = ax_full.imshow(frame_for_display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_full.plot([x], [y], marker=reticle(which='rt'), ms=15, color="cyan")
    ax_full.coords.grid(True, color="white", alpha=0.35, ls=":")
    ax_full.set_xlabel("Right Ascension")
    ax_full.set_ylabel("Declination")
    ax_full.set_title(f"Full image @ t={times[t_full_center]:.3f}s (idx {t_full_center})")
    cbar1 = fig.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)
    cbar1.set_label("Intensity (arb. units)")
    
    # Top-right: WCS cutout
    ax_cut = fig.add_subplot(2, 2, 2, projection=wcs_cut)
    im_cut = ax_cut.imshow(snippet, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_cut.plot([min(half_sp, x - x0)], [min(half_sp, y - y0)], marker=reticle(which='rt'), ms=15, color="cyan")
    ax_cut.coords.grid(True, color="white", alpha=0.35, ls=":")
    ax_cut.set_xlabel("Right Ascension")
    ax_cut.set_ylabel("Declination")
    ax_cut.set_title(f"Cutout")


    cbar2 = fig.colorbar(im_cut, ax=ax_cut, fraction=0.046, pad=0.04)
    cbar2.set_label("Intensity (arb. units)")    


    # Middle: full-res light curve
    ax_lc_full = fig.add_subplot(3, 1, 2)  # spans full width below
    ax_lc_full.plot(times, lc_full, color="tab:blue", lw=1.6)
    ax_lc_full.axvline(times[t_full_center], color="red", ls="--", lw=1.0, label="Detection time")
    ax_lc_full.set_ylabel("Flux (full-res)")
    ax_lc_full.grid(True, alpha=0.3)
    ax_lc_full.legend(loc="best")

    # Bottom: boxcar-smoothed light curve
    ax_lc_box = fig.add_subplot(3, 1, 3)
    if T_eff > 0:
        ax_lc_box.plot(times_sm, lc_sm, color="tab:green", lw=1.6)
        ax_lc_box.axvline(times_sm[k_center], color="red", ls="--", lw=1.0, label=f"Smoothed center (w={w})")
        ax_lc_box.legend(loc="best")
    else:
        ax_lc_box.text(0.5, 0.5, f"Width {w} > T; no smoothed LC",
                       transform=ax_lc_box.transAxes, ha="center", va="center", color="red")
    ax_lc_box.set_xlabel("Time (s)")
    ax_lc_box.set_ylabel(f"Flux (boxcar mean, w={w})")
    ax_lc_box.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, wspace=0.22, hspace=0.35)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)

    return {"lc_full": out_full, "lc_boxcar": out_box, "figure": out_fig}


    
def save_candidate_snippet_products(snippet_rec: dict,
                                    out_prefix: str,
                                    *,
                                    pixscale_rad: float,
                                    ra_rad: float,
                                    dec_rad: float,
                                    ra_sign: int = -1,
                                    dec_sign: int = 1,
                                    cmap: str = "gray",
                                    gif_fps: int = 5,
                                    dpi: int = 300) -> dict:
    """
    Save three products for a transient candidate snippet:
      1) animated GIF across time with RA/Dec labels (WCSAxes)
      2) static PNG at the detection frame + light curve of detection pixel
      3) FITS image of the detection frame with TAN WCS

    Parameters
    ----------
    snippet_rec : dict
        An entry from `extract_candidate_snippets(...)`:
        { "candidate": {...}, "snippet_cube": (T, N, N) array, "snippet_times": (T,) array, ... }
        The detection pixel is at the spatial center (N//2, N//2).
    out_prefix : str
        Prefix for output file names (e.g., '/path/run001_cand0001').
    pixscale_rad : float
        Pixel scale in radians per pixel (ducc wgridder input).
    ra_rad, dec_rad : float
        World coordinates (radians) of the detection pixel.
    ra_sign : int
        Sign for RA pixel scale in WCS (default -1 for standard leftward RA).
    cmap : str
        Matplotlib colormap for images.
    gif_fps : int
        Frames per second for GIF.
    dpi : int
        DPI for saved PNG/GIF frames.

    Returns
    -------
    dict with paths:
        { "gif": ..., "png": ..., "fits": ... }
    """
    cube = snippet_rec["snippet_cube"]  # (T, N, N)
    times = snippet_rec["snippet_times"]  # (T,)
    cand  = snippet_rec.get("candidate", {})
    T, N, M = cube.shape
    assert N == M, "snippet must be square"

    # Detection frame inside the snippet: the temporal center
    t_center = T // 2
    y_center = N // 2
    x_center = M // 2

    # Build WCS for RA/Dec labels

    wcs2d = ducc_wcs._build_tan_wcs_for_snippet(
        spatial_size=N,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        pixscale_rad=pixscale_rad,
        ra_sign=ra_sign,           # leftward RA increases (astronomical convention)
        dec_sign=dec_sign,
        radesys="ICRS",       # or "FK5" if that matches dataset
        equinox=None          # set 2000.0 if using FK5
    )
    
    # Robust display scaling
    det_frame = cube[t_center]
    vmin = np.nanpercentile(det_frame, 5.0)
    vmax = np.nanpercentile(det_frame, 99.5)

    out_gif  = f"{out_prefix}_snippet.gif"
    out_png  = f"{out_prefix}_det.png"
    out_fits = f"{out_prefix}_det.fits"

    # 1) Animated GIF with WCSAxes
    fig = plt.figure(figsize=(5, 4), dpi=dpi)
    ax = fig.add_subplot(111, projection=wcs2d)
    im = ax.imshow(cube[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    # Sky grid and labels
    ax.coords.grid(True, color="white", alpha=0.3, ls=":")
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    # Mark the detection pixel at center
    ax.plot([x_center], [y_center], marker=reticle(which='rt'), ms=4, color="cyan")
    # Title updates with time
    def _update(k: int):
        im.set_data(cube[k])
        t = times[k]
        if np.isfinite(t):
            ax.set_title(f"t = {t:.2f} s")
        else:
            ax.set_title("t = (pad)")
        return (im,)
    anim = FuncAnimation(fig, _update, frames=T, interval=1000//gif_fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=gif_fps))
    plt.close(fig)

    # 2) Static PNG at detection frame + light curve
    fig2 = plt.figure(figsize=(6, 6), dpi=dpi)
    # Top: WCSAxes image
    ax_img = fig2.add_subplot(2, 1, 1, projection=wcs2d)
    im2 = ax_img.imshow(det_frame, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_img.coords.grid(True, color="white", alpha=0.3, ls=":")
    ax_img.set_xlabel("Right Ascension")
    ax_img.set_ylabel("Declination")
    ax_img.plot([x_center], [y_center], marker=reticle(which='rt'), ms=15, color="cyan")
    cb = fig2.colorbar(im2, ax=ax_img, fraction=0.046, pad=0.04)
    cb.set_label("Intensity (arb. units)")
    # Title with RA/Dec, SNR if available
    ra_hms = cand.get("ra_hms", None)
    dec_dms = cand.get("dec_dms", None)
    title_txt = "Detection frame"
    if ra_hms and dec_dms:
        title_txt += f"  |  RA={ra_hms}  Dec={dec_dms}"
    ax_img.set_title(title_txt)

    # Bottom: light curve of detection pixel
    ax_lc = fig2.add_subplot(2, 1, 2)
    lc = cube[:, y_center, x_center]
    # Mask NaNs from pad zones
    good = np.isfinite(times) & np.isfinite(lc)
    ax_lc.plot(times[good], lc[good], color="tab:blue", lw=1.5)
    # Mark the detection center time
    t0 = times[t_center] if np.isfinite(times[t_center]) else np.nan

    print(cand)
    try:
        tline = cand.get("time_center_peak", cand["time_center"])
        ax_lc.axvline(tline, color="red", ls="--", lw=1.0, label="Detection (peak)")

    except KeyError:
        print("time_center not found")
        
    # if np.isfinite(t0):
    #     ax_lc.axvline(t0, color="red", ls="--", lw=1.0, label="Detection time")
    #     ax_lc.legend(loc="best")
    ax_lc.set_xlabel("Time (s)")
    ax_lc.set_ylabel("Pixel value at detection location")
    ax_lc.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(out_png, bbox_inches="tight")
    plt.close(fig2)

    # 3) FITS image of detection frame with WCS header

    hdr = wcs2d.to_header()
    
    # Optional helpful metadata:
    hdr["BUNIT"] = "arb. unit"
    hdr["COMMENT"] = "Detection frame from transient snippet"
    if ra_hms and dec_dms:
        hdr["OBJRA"] = ra_hms
        hdr["OBJDEC"] = dec_dms
        
    # Ensure CDELT/PC are present (to_header() should already include them,
    # but we can set explicitly to be safe):
    # NOTE: when using CDELT+PC we DO NOT set CD.
    # CDELT1/2 in degrees per pixel
    hdr["CDELT1"] = wcs2d.wcs.cdelt[0]
    hdr["CDELT2"] = wcs2d.wcs.cdelt[1]
    
    # Identity PC
    hdr["PC1_1"] = 1.0; hdr["PC1_2"] = 0.0
    hdr["PC2_1"] = 0.0; hdr["PC2_2"] = 1.0
    
    # Frame metadata if present
    if getattr(wcs2d.wcs, "radesys", None):
        hdr["RADESYS"] = wcs2d.wcs.radesys
    #if getattr(wcs2d.wcs, "equinox", None) is not None:
    #    hdr["EQUINOX"] = wcs2d.wcs.equinox
        

    # CTYPE/CUNIT/CRPIX/CRVAL are already provided by to_header(), so no need to duplicate.
    
    hdu = fits.PrimaryHDU(data=det_frame.astype(np.float32), header=hdr)
    hdu.writeto(out_fits, overwrite=True)

    return {"gif": out_gif, "png": out_png, "fits": out_fits}


def _stack_tables_or_empty(tables):
    """Vstack a list of tables (outer join), return a filled table or empty schema."""
    if len(tables) == 0:
        # Create a minimal empty table with common columns used in your pipeline
        return Table(names=["x","y","l","m","ra_rad","dec_rad","ra_deg","dec_deg",
                            "ra_hms","dec_dms","snr","std","width_samples",
                            "time_start","time_end","time_center","t0_idx","t1_idx_excl",
                            "center_idx","phase_center_field","chunk_id","algo"],
                     dtype=[int,int,float,float,float,float,float,float,
                            "U20","U20",float,float,int,
                            float,float,float,int,int,
                            int,"U64",int,"U16"])
    # Use outer join to be resilient to column differences across chunks
    T = vstack(tables, join_type="outer", metadata_conflicts="silent")
    # Fill masked values (NA) with sensible defaults for CSV/VOT output
    T.fill_value = np.nan
    T = T.filled()
    return T


def _read_csv_tables(paths):
    """Read a list of CSV files into Astropy Tables (skip missing)."""
    tables = []
    for p in paths:
        try:
            t = Table.read(p, format="csv")
            tables.append(t)
        except Exception as e:
            print(f"[Consolidation] Skipping '{p}' (read error: {e})")
    return tables

def _stack_tables_or_empty(tables):
    """Vstack a list of tables (outer join), return a filled table or empty schema."""
    if len(tables) == 0:
        # Create a minimal empty table with common columns used in your pipeline
        return Table(names=["x","y","l","m","ra_rad","dec_rad","ra_deg","dec_deg",
                            "ra_hms","dec_dms","snr","std","width_samples",
                            "time_start","time_end","time_center","t0_idx","t1_idx_excl",
                            "center_idx","phase_center_field","chunk_id","algo"],
                     dtype=[int,int,float,float,float,float,float,float,
                            "U20","U20",float,float,int,
                            float,float,float,int,int,
                            int,"U64",int,"U16"])
    # Use outer join to be resilient to column differences across chunks
    T = vstack(tables, join_type="outer", metadata_conflicts="silent")
    # Fill masked values (NA) with sensible defaults for CSV/VOT output
    T.fill_value = np.nan
    T = T.filled()
    return T


def consolidate_chunk_catalogues(
    *,
    ms_base: str,                         # e.g., "example" from "example.ms"
    out_dir: str,                         # e.g., "<MS parent>/candidates"
    var_csv_pattern: str,                 # e.g., "<candidates>/<ms_base>_chunk_*_var_candidates.csv"
    box_csv_pattern: str,                 # e.g., "<candidates>/<ms_base>_chunk_*_boxcar_candidates.csv"
    remove_chunk_catalogues: bool = True
):
    """
    Consolidate per-chunk catalogues from disk into MS-local `candidates/`,
    write consolidated CSV+VOT with names like:
        <out_dir>/<ms_base>_variance_all.csv
        <out_dir>/<ms_base>_variance_all.vot
        <out_dir>/<ms_base>_boxcar_all.csv
        <out_dir>/<ms_base>_boxcar_all.vot
    Optionally remove per-chunk CSV/VOT files afterwards.

    Parameters
    ----------
    ms_base : str
        Base name of the measurement set (e.g., "example" for "example.ms").
    out_dir : str
        Destination directory (the `candidates/` folder next to the MS).
    var_csv_pattern : str
        Glob pattern for variance per-chunk CSV files.
        Example: os.path.join(candidates_dir, f"{ms_base}_chunk_*_var_candidates.csv")
    box_csv_pattern : str
        Glob pattern for boxcar per-chunk CSV files.
        Example: os.path.join(candidates_dir, f"{ms_base}_chunk_*_boxcar_candidates.csv")
    remove_chunk_catalogues : bool
        If True, delete per-chunk CSV/VOT files after consolidation.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Glob per-chunk CSV files
    var_csv_files = sorted(glob.glob(var_csv_pattern))
    box_csv_files = sorted(glob.glob(box_csv_pattern))

    print(f"[Consolidation] Found {len(var_csv_files)} variance CSVs and {len(box_csv_files)} boxcar CSVs.")

    # 2) Read and stack
    var_tables = _read_csv_tables(var_csv_files)
    box_tables = _read_csv_tables(box_csv_files)

    t_var_all = _stack_tables_or_empty(var_tables)
    t_box_all = _stack_tables_or_empty(box_tables)

    # 3) Write consolidated catalogues (CSV + VOT) into candidates dir with ms_base prefix
    var_out_csv = os.path.join(out_dir, f"{ms_base}_variance_all.csv")
    var_out_vot = os.path.join(out_dir, f"{ms_base}_variance_all.vot")
    t_var_all.write(var_out_csv, format="csv", overwrite=True)
    t_var_all.write(var_out_vot, format="votable", overwrite=True)

    box_out_csv = os.path.join(out_dir, f"{ms_base}_boxcar_all.csv")
    box_out_vot = os.path.join(out_dir, f"{ms_base}_boxcar_all.vot")
    t_box_all.write(box_out_csv, format="csv", overwrite=True)
    t_box_all.write(box_out_vot, format="votable", overwrite=True)

    print(f"[Consolidation] Wrote {len(t_var_all)} variance candidates -> {var_out_csv}, {var_out_vot}")
    print(f"[Consolidation] Wrote {len(t_box_all)} boxcar candidates   -> {box_out_csv}, {box_out_vot}")

    # 4) Remove per-chunk catalogues (CSV + VOT) if requested
    if remove_chunk_catalogues:
        var_vot_files = sorted(glob.glob(var_csv_pattern.replace(".csv", ".vot")))
        box_vot_files = sorted(glob.glob(box_csv_pattern.replace(".csv", ".vot")))
        removed = 0
        for p in (var_csv_files + var_vot_files + box_csv_files + box_vot_files):
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                print(f"[Consolidation] Could not remove '{p}': {e}")
        print(f"[Consolidation] Removed {removed} per-chunk catalogue files.")
