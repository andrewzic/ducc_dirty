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

from fastducc import wcs as ducc_wcs
from fastducc import filters, kernels, candidates, ms_utils, detection

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
            S = kernels._moving_sum_from_csum(csum, w)  # (T_eff, Ny, Nx)
            T_eff = S.shape[0]

            # Compute a spatial sigma per time window
            if spatial_estimator == "mad":
                sigma_w = kernels._mad_std_2d_per_time(S, valid_mask)  # (T_eff,)
            elif spatial_estimator == "clipped_rms":
                sigma_w = kernels._clipped_rms_2d_per_time(S, valid_mask, clip_sigma)
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
            snr = kernels._temporal_std_snr(data, csum, csum2, w)  # (T_eff, Ny, Nx)
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

            center_idx  = t0 + (w // 2)
            time_center = float(times[center_idx])

            # # nearest index to time_center, constrained to [t0, t1-1]
            # k = np.searchsorted(times[t0:t1], time_center)
            # if k == 0:
            #     center_idx = t0
            # elif k >= (t1 - t0):
            #     center_idx = t1 - 1
            # else:
            #     left = t0 + (k - 1)
            #     right = t0 + k
            #     center_idx = left if abs(times[left] - time_center) <= abs(times[right] - time_center) else right

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
                    kernels._moving_sum_from_csum(csum, w)[t0, ys[i], xs[i]]
                ),
            }
            detections.append(det)

    return detections, snr_cubes


def variance_search(
    times: np.ndarray,
    cube: np.ndarray,                # (T, Ny, Nx)
    *,
    threshold_sigma: float = 5.0,
    return_snr_cubes: bool = False,
    keep_top_k: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,
    subtract_mean_per_pixel: bool = False,
    spatial_estimator: str = "clipped_rms",  # "mad" | "clipped_rms"
    clip_sigma: float = 3.0,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """
    Detect variability by the per-pixel standard deviation across time.

    Steps:
      1) Optionally subtract the temporal mean per pixel (high-pass).
      2) Compute per-pixel std along time: std_map = std(data, axis=0).
      3) Estimate a global spatial sigma from std_map using MAD or sigma-clipped RMS.
      4) Form SNR map: snr = std_map / spatial_sigma.
      5) Threshold and (optionally) keep the top-K detections.

    Parameters
    ----------
    times : (T,) ndarray
        Time stamps for the cube. Only length consistency is checked here.
    cube : (T, Ny, Nx) ndarray
        Full-resolution image cube (float-like).
    threshold_sigma : float
        SNR threshold applied to the 2-D map of std-derived SNR.
    return_snr_cubes : bool
        If True, returns a dict containing the 2-D SNR map under key "std".
    keep_top_k : Optional[int]
        If set, retain only the top-K detections by SNR across the image.
    valid_mask : Optional[(Ny, Nx) ndarray of bool]
        Mask of valid pixels. True=valid. If None, all pixels are valid.
    subtract_mean_per_pixel : bool
        If True, subtract per-pixel mean across time before computing std.
        This makes the std sensitive to variability rather than steady flux.
    spatial_estimator : {"mad", "clipped_rms"}
        Robust spatial estimator used to convert std_map to an SNR map.
    clip_sigma : float
        Sigma parameter for the sigma-clipped RMS estimator.

    Returns
    -------
    detections : list of dict
        Each dict contains:
          {
            "y": int,
            "x": int,
            "snr": float,
            "std": float,           # the per-pixel std at (y,x)
            "time_start": None,     # not used here; set to None for completeness
            "time_end": None,
            "time_center": None,
            "center_idx": None,
          }
        (No references to window indices or t0.)

    snr_cubes : Optional[dict]
        If return_snr_cubes is True: {"std": snr_2d [Ny, Nx] (float32)}.
        Otherwise, None.
    """
    # --- validate inputs ---
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

    # Prepare data (optional temporal mean subtraction)
    data = cube.astype(np.float64, copy=False)
    if subtract_mean_per_pixel:
        mean_map = np.nanmean(data, axis=0, keepdims=True)
        data = data - mean_map

    # 1) per-pixel std over time
    std_map = np.nanstd(data, axis=0)

    # 2) robust spatial sigma for std_map
    if spatial_estimator == "mad":
        spatial_sigma = kernels._mad_sigma_2d(std_map, valid_mask)
    elif spatial_estimator == "clipped_rms":
        spatial_sigma = kernels._clipped_rms_sigma_2d(std_map, valid_mask, clip_sigma=clip_sigma)
    else:
        raise ValueError(f"Unknown spatial_estimator='{spatial_estimator}'")

    # 3) SNR map (2-D)
    #if not np.isfinite(spatial_sigma) or spatial_sigma <= 0:
    #spatial_sigma = 1.0
    snr = (std_map - np.nanmean(std_map)) / spatial_sigma

    # 4) apply mask (gate invalid pixels to 0)
    snr = np.where(valid_mask, snr, 0.0)

    # 5) threshold
    ys, xs = np.where(snr >= threshold_sigma)
    if ys.size == 0:
        detections: List[Dict[str, Any]] = []
        snr_out = {"std": snr.astype(np.float32, copy=False)} if return_snr_cubes else None
        return detections, snr_out

    # 6) optional Top-K: by descending SNR
    snr_vals = snr[ys, xs]
    order = np.argsort(snr_vals)[::-1]
    if keep_top_k is not None and ys.size > keep_top_k:
        order = order[:keep_top_k]
    ys = ys[order]
    xs = xs[order]
    snr_vals = snr_vals[order]

    # 7) pack detections (no window/time indices here)
    detections: List[Dict[str, Any]] = []
    for y, x, s in zip(ys, xs, snr_vals):
        det = {
            "y": int(y),
            "x": int(x),
            "snr": float(s),
            "std": float(std_map[y, x]),
            # placeholders for API consistency (unused in variance search)
            "time_start": None,
            "time_end": None,
            "time_center": None,
            "center_idx": None,
        }
        detections.append(det)

    snr_cubes = {"std": snr.astype(np.float32, copy=False)} if return_snr_cubes else None
    return detections, snr_cubes


def variance_search_welford(
    std_map: np.ndarray,                      # (Ny, Nx) final (or partial) Welford std map
    *,
    threshold_sigma: float = 5.0,
    return_snr_image: bool = False,           # renamed for clarity
    keep_top_k: Optional[int] = None,
    valid_mask: Optional[np.ndarray] = None,  # (Ny, Nx) True=valid
    spatial_estimator: str = "clipped_rms",   # "mad" | "clipped_rms"
    clip_sigma: float = 3.0,
    subtract_mean_of_std_map: bool = True    # if True, SNR = (std - mean(std)) / sigma
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Detect variability from a precomputed per-pixel standard deviation map (Welford result).

    Steps:
      1) Robustly estimate a spatial sigma on std_map (MAD or clipped RMS).
      2) Form SNR map: snr = std_map / spatial_sigma (or mean-subtracted variant).
      3) Apply mask, threshold, optional Top-K.
      4) Return detections and (optionally) the SNR image.

    Parameters
    ----------
    std_map : (Ny, Nx) ndarray
        Per-pixel standard deviation across the *full observation* computed via Welford.
    threshold_sigma : float
        SNR threshold applied to the 2-D SNR map.
    return_snr_image : bool
        If True, returns the SNR image (float32) as the second element of the tuple.
    keep_top_k : Optional[int]
        If set, retain only the top-K detections by SNR across the image.
    valid_mask : Optional[(Ny, Nx) ndarray of bool]
        Mask of valid pixels. True=valid. If None, all pixels are valid.
    spatial_estimator : {"mad", "clipped_rms"}
        Robust spatial estimator used to convert std_map to an SNR map.
    clip_sigma : float
        Sigma parameter for the sigma-clipped RMS estimator.
    subtract_mean_of_std_map : bool
        If True, SNR is computed as (std_map - mean(std_map_valid)) / spatial_sigma.
        If False, SNR is std_map / spatial_sigma.

    Returns
    -------
    detections : list of dict
        Each dict contains:
          {
            "y": int,
            "x": int,
            "snr": float,
            "std": float,           # the per-pixel std at (y,x)
            "time_start": None,     # placeholders for API compatibility
            "time_end": None,
            "time_center": None,
            "center_idx": None,
            "width_samples": 1      # variance search is window-less.
          }
    snr_image : Optional[np.ndarray]
        If return_snr_image is True: the 2-D SNR image (Ny, Nx), dtype float32.
        Otherwise, None.
    """
    # --- validate inputs ---
    if std_map.ndim != 2:
        raise ValueError("std_map must be (Ny, Nx)")
    Ny, Nx = std_map.shape

    if valid_mask is None:
        valid_mask = np.ones((Ny, Nx), dtype=bool)
    elif valid_mask.shape != (Ny, Nx):
        raise ValueError("valid_mask must be shape (Ny, Nx)")

    # --- robust spatial sigma for std_map ---
    std_map64 = std_map.astype(np.float64, copy=False)
    if spatial_estimator == "mad":
        spatial_sigma = kernels._mad_sigma_2d(std_map64, valid_mask.astype(np.bool_, copy=False))
    elif spatial_estimator == "clipped_rms":
        spatial_sigma = kernels._clipped_rms_sigma_2d(std_map64, valid_mask.astype(np.bool_, copy=False),
                                              clip_sigma=clip_sigma)
    else:
        raise ValueError(f"Unknown spatial_estimator='{spatial_estimator}'")

    if (not np.isfinite(spatial_sigma)) or (spatial_sigma <= 0.0):
        spatial_sigma = 1.0

    # --- SNR map: mean-subtracted or direct ratio ---
    if subtract_mean_of_std_map:
        mu = np.nanmean(std_map64)
        snr = (std_map64 - mu) / spatial_sigma
    else:
        snr = std_map64 / spatial_sigma

    # --- gate invalid pixels ---
    snr = np.where(valid_mask, snr, 0.0)

    # --- threshold ---
    ys, xs = np.where(snr >= threshold_sigma)
    if ys.size == 0:
        detections: List[Dict[str, Any]] = []
        snr_out = snr.astype(np.float32, copy=False) if return_snr_image else None
        return detections, snr_out

    # --- optional Top-K by descending SNR ---
    snr_vals = snr[ys, xs]
    order = np.argsort(snr_vals)[::-1]
    if (keep_top_k is not None) and (ys.size > keep_top_k):
        order = order[:keep_top_k]
    ys = ys[order]
    xs = xs[order]
    snr_vals = snr_vals[order]

    # --- pack detections with boxcar-style keys for downstream API compatibility ---
    detections: List[Dict[str, Any]] = []
    for y, x, s in zip(ys, xs, snr_vals):
        det = {
            "y": int(y),
            "x": int(x),
            "snr": float(s),
            "std": float(std_map64[y, x]),
            # placeholders to satisfy downstream code
            "time_start": 0,
            "time_end": 0,
            "time_center": 0,
            "center_idx": 0,
            "width_samples": 1,
        }
        detections.append(det)

    snr_out = snr.astype(np.float32, copy=False) if return_snr_image else None
    return detections, snr_out
