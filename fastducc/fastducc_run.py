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
from fastducc.types import Config, WelfordState
from fastducc import filters, kernels, candidates, ms_utils, detection, imaging

def build_cli():
    
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
    parser.add_argument('--boxcar-widths', nargs='+', type=int,
                        help='Widths (in samples) to perform boxcar search over (default: [1, 2, 4, 8, 16, 32, 64])',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--threshold-sigma', type=float, default=8.0, help="S/N threshold to use for detections (default: 8.0)")
    parser.add_argument('--do-plot', action='store_true')
    parser.add_argument('--var-threshold-sigma', type=float, default=8.0)
    parser.add_argument('--var-keep-top-k', type=int, default=None)
    parser.add_argument('--var-clip-sigma', type=float, default=3.0)
    parser.add_argument('--var-nms-radius', type=int, default=10)
    # --- Search toggles ---
    parser.add_argument(
        '--enable-boxcar', dest='enable_boxcar', action='store_true',
        help='Enable per-width boxcar transient search (default: enabled)'
    )
    parser.add_argument(
        '--disable-boxcar', dest='enable_boxcar', action='store_false',
        help='Disable per-width boxcar transient search'
    )
    parser.set_defaults(enable_boxcar=True)
    parser.add_argument(
        '--enable-var', dest='enable_var', action='store_true',
        help='Enable variance (std-map/Welford) variability search (default: enabled)'
    )
    parser.add_argument(
        '--disable-var', dest='enable_var', action='store_false',
        help='Disable variance (std-map/Welford) variability search'
    )
    parser.set_defaults(enable_var=True)
    parser.add_argument(
        '--enable-var-chunk', dest='enable_var_chunk', action='store_true',
        help='Enable variance (std-map/Welford) variability search per chunk (default: disabled)'
    )
    parser.add_argument(
        '--disable-var-chunk', dest='enable_var_chunk', action='store_false',
        help='Disable variance (std-map/Welford) variability search per chunk'
    )
    parser.set_defaults(enable_var_chunk=False)
    parser.add_argument('--save-var-lightcurves',  dest='save_var_lightcurves',  action='store_true', help='Save lightcurves for variance candidates')
    parser.add_argument('--no-save-var-lightcurves',  dest='save_var_lightcurves',  action='store_false')
    parser.set_defaults(save_var_lightcurves=True)    
    parser.add_argument('--save-var-snippets',     dest='save_var_snippets',     action='store_true', help='Save snippet products (PNG/GIF/FITS) for variance candidates')
    parser.add_argument('--no-save-var-snippets',  dest='save_var_snippets',     action='store_false')
    parser.set_defaults(save_var_snippets=True)    
    parser.add_argument('--save-box-lightcurves',  dest='save_box_lightcurves',  action='store_true', help='Save lightcurves for boxcar candidates')
    parser.add_argument('--no-save-box-lightcurves',  dest='save_box_lightcurves',  action='store_false')
    parser.set_defaults(save_box_lightcurves=True)    
    parser.add_argument('--save-box-snippets',     dest='save_box_snippets',     action='store_true', help='Save snippet products (PNG/GIF/FITS) for boxcar candidates')
    parser.add_argument('--no-save-box-snippets',  dest='save_box_snippets',     action='store_false')
    parser.set_defaults(save_box_snippets=True)
    
    # --- Parallel execution flags ---   
    parser.add_argument(
        '--parallel-mode',
        choices=['serial', 'dask-local', 'dask-slurm'],
        default='serial',
        help='Serial, Dask LocalCluster, or Dask SLURMCluster execution'
    )
    parser.add_argument(
        '--dask-workers', type=int, default=0,
        help='Number of Dask workers when --parallel-mode=dask; 0 => auto'
    )
    parser.add_argument(
        '--threads-per-worker', type=int, default=1,
        help='Threads per Dask worker (default: 1)'
    )
    parser.add_argument(
        '--dask-scheduler', choices=['processes', 'threads'], default='processes',
        help='Use multiprocessing or multithreading workers (default: processes)'
    )
    # --- SLURM-only cluster params ---
    parser.add_argument('--slurm-partition', default=None, help='SLURM partition/queue')
    parser.add_argument('--slurm-account',  default=None, help='SLURM account/project')
    parser.add_argument('--slurm-cores-per-worker', type=int, default=1)
    parser.add_argument('--slurm-mem', default='8GB', help='Memory per worker (e.g., 8GB)')
    parser.add_argument('--slurm-walltime', default='01:00:00')
    parser.add_argument('--slurm-job-extra', nargs='*', default=[],
                        help='Extra SLURM directives, e.g. ["--exclusive"]')
    parser.add_argument('--slurm-interface', default=None,
                        help='Network interface name (if needed for TCP comms)')

    
    args = parser.parse_args()
    
    return args

def make_config(args, paths) -> Config:
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = paths
    ra0_rad, dec0_rad, _ = ms_utils.get_phase_center(args.msname, field_name=None)
    pix_rad = args.pixsize_arcsec / 206265.0
    return Config(
        msname=args.msname,
        npix_x=args.npix_x, npix_y=args.npix_y,
        pix_rad=pix_rad, ra0_rad=ra0_rad, dec0_rad=dec0_rad,
        epsilon=args.epsilon, do_wgridding=args.do_wgridding,
        nthreads=args.nthreads, verbosity=args.verbosity,
        corr_mode=args.corr_mode, basis=args.basis, single_pol=args.single_pol,
        data_column=args.data_column,
        enable_var=args.enable_var, enable_boxcar=args.enable_boxcar,
        enable_var_chunk=args.enable_var_chunk,
        save_var_lightcurves=args.save_var_lightcurves,
        save_var_snippets=args.save_var_snippets,
        save_box_lightcurves=args.save_box_lightcurves,
        save_box_snippets=args.save_box_snippets,
        var_threshold=args.var_threshold_sigma, var_keep_k=args.var_keep_top_k,
        var_clip_sigma=args.var_clip_sigma, var_nms_radius=args.var_nms_radius,
        boxcar_widths=args.boxcar_widths, boxcar_threshold=args.threshold_sigma,
        do_plot=args.do_plot,
        ms_base=ms_base, candidates_dir=candidates_dir,
        chunk_prefix_root=chunk_prefix_root, all_prefix_root=all_prefix_root
    )

def init_welford(cfg: Config) -> WelfordState:
    Ny, Nx = cfg.npix_y, cfg.npix_x
    return WelfordState(
        count=np.zeros((Ny, Nx), dtype=np.int64),
        mean=np.zeros((Ny, Nx), dtype=np.float64),
        M2=np.zeros((Ny, Nx), dtype=np.float64)
    )

def process_variance_chunk(cfg: Config, times, cube, wf: WelfordState, start_idx: int):
    # Accumulate Welford
    kernels.welford_update_cube(wf.count, wf.mean, wf.M2, cube, ignore_nan=True)
    if not cfg.enable_var:
        return []  # no candidates in this chunk

    if not cfg.enable_var_chunk:
        return []

    # Partial std-map (optional, for visuals or chunk-level variance search)
    std_map_partial = kernels.welford_finalise_std(wf.count, wf.M2, ddof=1)

    # Run Welford-based variance search on partial map
    var_dets, snr_img = detection.variance_search_welford(
        std_map_partial,
        threshold_sigma=cfg.var_threshold,
        return_snr_image=True,
        keep_top_k=cfg.var_keep_k,
        valid_mask=None,
        spatial_estimator="clipped_rms",
        clip_sigma=cfg.var_clip_sigma,
        subtract_mean_of_std_map=True
    )

    # Spatial NMS + annotation
    var_nms = filters.nms_snr_map_2d(
        snr_2d=snr_img, base_detections=var_dets,
        threshold_sigma=cfg.var_threshold,
        spatial_radius=cfg.var_nms_radius,
        valid_mask=None,
        times=times, cube=cube, time_tag_policy="peak_absdev"
    )
    annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
        msname=cfg.msname, final_detections=var_nms,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        flip_u=True, flip_v=True, field_name=None
    )

    # Save per-chunk VAR catalogue
    var_root = cfg.chunk_prefix_root(start_idx) + "_var"
    t_var = candidates.candidates_to_astropy_table(annotated_var)
    candidates.save_candidates_table(t_var,
        csv_path=f"{var_root}_candidates.csv",
        vot_path=f"{var_root}_candidates.vot"
    )


   # 7) Lightcurves + snippet products for each VAR candidate (using std map)
    for i, cand in enumerate(annotated_var):
        # Lightcurves figure (top panels use std-map images)
        if cfg.save_var_lightcurves:
            _ = candidates.save_candidate_lightcurves(
                times=times, cube=cube, candidate=cand,
                out_prefix=f"{var_root}_cand_{i:03d}_lc",
                spatial_size=50, save_format="npz",
                center_policy="right", cmap="gray", dpi=180,
                # WCS / scale
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                pix_rad=cfg.pix_rad,
                ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                # Draw std-map images on the top panels:
                std_map=std_map_partial, use_std_images=True
            )

        # Snippet products from std-map (time length = 1)
        if cfg.save_var_snippets:
            std_snip = candidates.make_stdmap_snippet(std_map_partial, cand, spatial_size=50)
            _ = candidates.save_candidate_snippet_products(
                snippet_rec=std_snip,
                out_prefix=f"{var_root}_cand_{i:03d}_snip",
                pixscale_rad=cfg.pix_rad,
                ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
            )    
    return annotated_var

def process_boxcar_chunk(cfg: Config, times, cube, start_idx: int):
    if not cfg.enable_boxcar:
        return []
    dets, snr_cubes = detection.boxcar_search_time(
        times, cube,
        widths=cfg.boxcar_widths,
        widths_in_seconds=False,
        threshold_sigma=cfg.boxcar_threshold,
        return_snr_cubes=True,
        keep_top_k=50,
        std_mode="spatial_per_window",
        subtract_mean_per_pixel=True
    )
    dets_by_w = filters.nms_snr_maps_per_width(
        snr_cubes, times,
        threshold_sigma=cfg.boxcar_threshold,
        spatial_radius=10,
        time_radius=2,
        valid_mask=None
    )
    final_dets = filters.group_filter_across_widths(
        dets_by_w, times,
        spatial_radius=10,
        time_radius=0,
        policy="max_snr",
        max_per_time_group=1,
        ny_nx=(cube.shape[1], cube.shape[2])
    )
    if len(final_dets) == 0:
        return []

    annotated = ducc_wcs.annotate_candidates_with_sky_coords(
        msname=cfg.msname, final_detections=final_dets,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        flip_u=True, flip_v=True, field_name=None
    )

    # Save per-chunk BOXCAR catalogue
    box_root = cfg.chunk_prefix_root(start_idx) + "_boxcar"
    t_box = candidates.candidates_to_astropy_table(annotated)
    candidates.save_candidates_table(t_box,
        csv_path=f"{box_root}_candidates.csv",
        vot_path=f"{box_root}_candidates.vot"
    )


    # Lightcurves + snippet products (boxcar)
    for i, cand in enumerate(annotated):
        # Lightcurves figure (top panels show the full-res frame)
        if cfg.save_box_lightcurves:
            _ = candidates.save_candidate_lightcurves(
                times=times, cube=cube, candidate=cand,
                out_prefix=f"{box_root}_cand_{i:03d}_lc",
                spatial_size=50, save_format="npz",
                center_policy="right", cmap="gray", dpi=180,
                # WCS / scale:
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                pix_rad=cfg.pix_rad,
                ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                # Use full-res images on top panels (std_map=None)
                std_map=None, use_std_images=False
            )

        # Extract smoothed snippet (your updated function that uses boxcar smoothing per width)
        if cfg.save_box_snippets:
            snippets = candidates.extract_candidate_snippets(
                times, cube, [cand],   # pass single cand to get one snippet
                spatial_size=50,
                time_factor=5,
                pad_mode="constant", pad_value=0.0,
                return_indices=True,
                center_policy="right"  # or "left" as you prefer
            )
            # There will be exactly one snippet for this cand
            snip = snippets[0]
            _ = candidates.save_candidate_snippet_products(
                snippet_rec=snip,
                out_prefix=f"{box_root}_cand_{i:03d}_snip",
                pixscale_rad=cfg.pix_rad,
                ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=6, dpi=180
            )
    return annotated


def welford_combine_aggregates(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    """
    Combine two Welford aggregates (array-wise) into one.
    All inputs are (Ny, Nx) arrays: count=int64, mean=float64, M2=float64.
    """
    total = count_a + count_b
    # Where no samples, keep zeros
    mask = (count_a > 0) & (count_b > 0)

    # Prepare outputs
    mean_out = mean_a.copy()
    M2_out = M2_a.copy()
    count_out = total

    if np.any(mask):
        # delta = mean_b - mean_a
        delta = np.zeros_like(mean_a)
        delta[mask] = mean_b[mask] - mean_a[mask]
        # mean = mean_a + delta * (count_b / total)
        mean_out[mask] = mean_a[mask] + delta[mask] * (count_b[mask] / total[mask])
        # M2 = M2_a + M2_b + delta^2 * count_a * count_b / total
        M2_out[mask] = (M2_a[mask] + M2_b[mask] +
                        (delta[mask] * delta[mask]) * (count_a[mask] * count_b[mask] / total[mask]))
    # Where only one side has samples, just take that side
    only_a = (count_a > 0) & (count_b == 0)
    mean_out[only_a] = mean_a[only_a]
    M2_out[only_a]   = M2_a[only_a]
    only_b = (count_b > 0) & (count_a == 0)
    mean_out[only_b] = mean_b[only_b]
    M2_out[only_b]   = M2_b[only_b]

    return count_out, mean_out, M2_out


def finalise_welford(cfg: Config, wf: WelfordState, times, cube):
    # Write full std-map (global)
    std_map_full = kernels.welford_finalise_std(wf.count, wf.M2, ddof=1)
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
        pixscale_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None
    )
    hdr = wcs_full.to_header()
    hdr["BUNIT"] = "std"; hdr["CDELT1"] = wcs_full.wcs.cdelt[0]; hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr["PC1_1"] = 1.0; hdr["PC1_2"] = 0.0; hdr["PC2_1"] = 0.0; hdr["PC2_2"] = 1.0
    full_std_fits = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_std_map_full.fits")
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)

    # Optional: run final variance detection on full std-map and save catalogue
    if cfg.enable_var:
        var_final, snr_img = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.var_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_final) > 0:
            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_final,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.var_nms_radius,
                valid_mask=None,
                times=times, cube=cube, time_tag_policy="peak_absdev"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = cfg.all_prefix_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                csv_path=f"{var_root}_candidates.csv",
                vot_path=f"{var_root}_candidates.vot"
            )


def finalise_welford_parallel(
        cfg: Config,
        agg_list: List,               # List[Tuple[np.ndarray, np.ndarray, np.ndarray]] as (count, mean, M2) per chunk
        *,
        run_final_variance: bool = True
):
    """
    Reduce per-chunk Welford aggregates into a full-observation std-map,
    write FITS with WCS, optionally run final variance search, and
    write the consolidated variance catalogue.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration (paths, WCS info, toggles, thresholds).
    agg_list : list of (times, cube, count, mean, M2)
        Per-chunk aggregates returned by the parallel chunk tasks.
        Each array is shape (Ny, Nx): count=int64, mean=float64, M2=float64.
    run_final_variance : bool
        If True and cfg.enable_var, run variance_search_welford on the full std-map.

    Returns
    -------
    std_map_full : np.ndarray
        Final global standard deviation map, shape (Ny, Nx), float64.
    """
    # --- 1) Reduce all per-chunk aggregates into one global aggregate ---
    Ny, Nx = cfg.npix_y, cfg.npix_x
    count_acc = np.zeros((Ny, Nx), dtype=np.int64)
    mean_acc  = np.zeros((Ny, Nx), dtype=np.float64)
    M2_acc    = np.zeros((Ny, Nx), dtype=np.float64)

    for (times, cube, c, m, M2) in agg_list:
        # combine per-chunk aggregate into the accumulator
        count_acc, mean_acc, M2_acc = welford_combine_aggregates(count_acc, mean_acc, M2_acc, c, m, M2)

    all_times = np.concatenate([a[0] for a in agg_list])
    order = np.argsort(all_times)
    all_times = all_times[order]
    all_cube = np.empty((int(np.sum(a[1].shape[0] for a in agg_list)), cfg.npix_y, cfg.npix_x), dtype=agg_list[0][1].dtype)
    start_ = 0
    for a in agg_list:
        nsub = a[1].shape[0]
        end_ = start_ + nsub
        all_cube[start_:end_] = a[1]
        start_ += nsub

    all_cube = all_cube[order]
    
    # --- 2) Finalise to full std-map ---
    std_map_full = kernels.welford_finalise_std(count_acc, M2_acc, ddof=1)

    # --- 3) Write full std-map FITS with full-frame WCS ---
    wcs_full = ducc_wcs._build_fullframe_wcs(
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
        pixscale_rad=cfg.pix_rad,
        ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None
    )
    hdr = wcs_full.to_header()
    hdr["BUNIT"]  = "std"
    hdr["CDELT1"] = wcs_full.wcs.cdelt[0]
    hdr["CDELT2"] = wcs_full.wcs.cdelt[1]
    hdr["PC1_1"]  = 1.0; hdr["PC1_2"] = 0.0
    hdr["PC2_1"]  = 0.0; hdr["PC2_2"] = 1.0

    full_std_fits = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_std_map_full.fits")
    fits.writeto(full_std_fits, data=std_map_full.astype(np.float32), header=hdr, overwrite=True)
    print(f"[Final] wrote full std-map -> {full_std_fits}")

    # --- 4) Optional final variance search on the full std-map ---
    if run_final_variance and cfg.enable_var:
        var_final, snr_img = detection.variance_search_welford(
            std_map_full,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.var_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_final) > 0:
            
            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_final,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.var_nms_radius,
                valid_mask=None,
                times=all_times, cube=all_cube, time_tag_policy="none"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = cfg.all_prefix_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            for i, cand in enumerate(annotated_var):
                if cfg.save_var_lightcurves:
                    candidates.save_candidate_lightcurves(
                        all_times, all_cube, cand,
                        out_prefix=f"{var_root}_cand_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=std_map_full, use_std_images=True
                    )
                if cfg.save_var_snippets:
                    std_snip = candidates.make_stdmap_snippet(std_map_full, cand, spatial_size=50)
                    candidates.save_candidate_snippet_products(
                        snippet_rec=std_snip,
                        out_prefix=f"{var_root}_cand_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
                    )

    return std_map_full

def finalise_welford_serial(cfg: Config, wf_state: WelfordState, *, run_final_variance=True):
    """
    Finalise from a live WelfordState (count, mean, M2) as in serial runs,
    using the same parallel finalise routine.
    """
    agg_list = [(wf_state.count, wf_state.mean, wf_state.M2)]
    return finalise_welford_parallel(cfg, agg_list, run_final_variance=run_final_variance)

            
def consolidate_catalogues(cfg: Config):
    var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_var_candidates.csv")
    box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_chunk_*_boxcar_candidates.csv")
    candidates.consolidate_chunk_catalogues(
        ms_base=cfg.ms_base,
        out_dir=cfg.candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )



def process_chunk_task(cfg: Config, ms_base: str, candidates_dir: str, start: int, end: int):
    """
    Worker task: image the chunk, produce per-chunk outputs, and return per-chunk Welford aggregates.
    """
    # Imaging: open MS inside worker (t_main=None)
    times, cube = imaging.image_time_samples(
        msname=cfg.msname, t_main=None,
        start_time_idx=start, end_time_idx=end,
        corr_mode=cfg.corr_mode, basis=cfg.basis, single_pol=cfg.single_pol,
        data_column=cfg.data_column,
        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
        pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
        epsilon=cfg.epsilon, do_wgridding=cfg.do_wgridding,
        nthreads=cfg.nthreads, verbosity=cfg.verbosity, do_plot=cfg.do_plot
    )

    # Per-chunk Welford aggregates
    Ny, Nx = cfg.npix_y, cfg.npix_x
    c = np.zeros((Ny, Nx), dtype=np.int64)
    m = np.zeros((Ny, Nx), dtype=np.float64)
    M2 = np.zeros((Ny, Nx), dtype=np.float64)
    kernels.welford_update_cube(c, m, M2, cube, ignore_nan=True)

    chunk_root = os.path.join(candidates_dir, f"{ms_base}_chunk_{start:06d}")

    # Optional per-chunk variance search
    if cfg.enable_var and cfg.enable_var_chunk:
        std_map_partial = kernels.welford_finalise_std(c, M2, ddof=1)
        var_dets, snr_img = detection.variance_search_welford(
            std_map_partial,
            threshold_sigma=cfg.var_threshold,
            return_snr_image=True,
            keep_top_k=cfg.var_keep_k,
            valid_mask=None,
            spatial_estimator="clipped_rms",
            clip_sigma=cfg.var_clip_sigma,
            subtract_mean_of_std_map=True
        )
        if len(var_dets) > 0:
            var_nms = filters.nms_snr_map_2d(
                snr_2d=snr_img, base_detections=var_dets,
                threshold_sigma=cfg.var_threshold,
                spatial_radius=cfg.var_nms_radius,
                valid_mask=None,
                times=times, cube=cube, time_tag_policy="peak_absdev"
            )
            annotated_var = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=var_nms,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            var_root = chunk_root + "_var"
            t_var = candidates.candidates_to_astropy_table(annotated_var)
            candidates.save_candidates_table(t_var,
                                             csv_path=f"{var_root}_candidates.csv",
                                             vot_path=f"{var_root}_candidates.vot"
                                             )
            for i, cand in enumerate(annotated_var):
                if cfg.save_var_lightcurves:
                    candidates.save_candidate_lightcurves(
                        times, cube, cand,
                        out_prefix=f"{var_root}_cand_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=std_map_partial, use_std_images=True
                    )
                if cfg.save_var_snippets:
                    std_snip = candidates.make_stdmap_snippet(std_map_partial, cand, spatial_size=50)
                    candidates.save_candidate_snippet_products(
                        snippet_rec=std_snip,
                        out_prefix=f"{var_root}_cand_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=1, dpi=180
                    )

    # Optional boxcar search
    if cfg.enable_boxcar:
        dets, snr_cubes = detection.boxcar_search_time(
            times, cube,
            widths=cfg.boxcar_widths,
            widths_in_seconds=False,
            threshold_sigma=cfg.boxcar_threshold,
            return_snr_cubes=True,
            keep_top_k=50,
            std_mode="spatial_per_window",
            subtract_mean_per_pixel=True
        )
        dets_by_w = filters.nms_snr_maps_per_width(
            snr_cubes, times,
            threshold_sigma=cfg.boxcar_threshold,
            spatial_radius=10, time_radius=2, valid_mask=None
        )
        final_dets = filters.group_filter_across_widths(
            dets_by_w, times,
            spatial_radius=10, time_radius=0,
            policy="max_snr", max_per_time_group=1,
            ny_nx=(cube.shape[1], cube.shape[2])
        )
        if len(final_dets) > 0:
            annotated = ducc_wcs.annotate_candidates_with_sky_coords(
                msname=cfg.msname, final_detections=final_dets,
                npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                pixsize_x=cfg.pix_rad, pixsize_y=cfg.pix_rad,
                flip_u=True, flip_v=True, field_name=None
            )
            box_root = chunk_root + "_boxcar"
            t_box = candidates.candidates_to_astropy_table(annotated)
            candidates.save_candidates_table(t_box,
                csv_path=f"{box_root}_candidates.csv",
                vot_path=f"{box_root}_candidates.vot"
            )
            for i, cand in enumerate(annotated):
                if cfg.save_box_lightcurves:
                    candidates.save_candidate_lightcurves(
                        times, cube, cand,
                        out_prefix=f"{box_root}_cand_{i:03d}_lc",
                        spatial_size=50, save_format="npz",
                        center_policy="right", cmap="gray", dpi=180,
                        npix_x=cfg.npix_x, npix_y=cfg.npix_y,
                        ra0_rad=cfg.ra0_rad, dec0_rad=cfg.dec0_rad,
                        pix_rad=cfg.pix_rad, ra_sign=-1, dec_sign=-1, radesys="ICRS", equinox=None,
                        std_map=None, use_std_images=False
                    )
                if cfg.save_box_snippets:
                    snippets = candidates.extract_candidate_snippets(
                        times, cube, [cand],
                        spatial_size=50, time_factor=5,
                        pad_mode="constant", pad_value=0.0,
                        return_indices=True, center_policy="right"
                    )
                    snip = snippets[0]
                    candidates.save_candidate_snippet_products(
                        snippet_rec=snip,
                        out_prefix=f"{box_root}_cand_{i:03d}_snip",
                        pixscale_rad=cfg.pix_rad,
                        ra_rad=float(cand["ra_rad"]), dec_rad=float(cand["dec_rad"]),
                        ra_sign=-1, dec_sign=-1, cmap="gray", gif_fps=6, dpi=180
                    )

    return times, cube, c, m, M2

def main_serial(args):
    #args = build_cli()

    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)
    cfg = make_config(args, (ms_base, candidates_dir, chunk_prefix_root, all_prefix_root))
    t_main, total_chunks, time_col = ms_utils.open_ms(args.msname)
    # Discover total number of time chunks via iter    
    print(f"Found {total_chunks} time chunks in MS: {args.msname}")
    
    # Convert arcsec to radians
    pix_rad = args.pixsize_arcsec / 206265.0
    ra0_rad, dec0_rad, used_field = ms_utils.get_phase_center(args.msname, field_name=None)
    
    start = 0
    chunk_size = max(1, args.chunk_size)
    chunk_id = 0

    wf = init_welford(cfg)
    
    #t_main = table(args.msname, readonly=True)
    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        
        print(f"[Chunk {chunk_id}] imaging time_idx {start}..{end}")
        times, cube = imaging.image_time_samples(msname=args.msname,
                                         t_main=t_main,
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

        
        # Variance/Welford chunk
        var_ann = process_variance_chunk(cfg, times, cube, wf, start)

        # Boxcar chunk
        box_ann = process_boxcar_chunk(cfg, times, cube, start)

        start = end + 1
        chunk_id += 1
        
    # Finalisation
    finalise_welford(cfg, wf, times, cube)
    consolidate_catalogues(cfg)



def main():
    args = build_cli()

    if args.parallel_mode == "serial":
        main_serial(args)
        return None
    
    ms_base, candidates_dir, chunk_prefix_root, all_prefix_root = ms_utils.derive_paths(args.msname)
    cfg = make_config(args, (ms_base, candidates_dir, chunk_prefix_root, all_prefix_root))

    # Compute chunk bounds (using casacore once in the driver)
    t_main = table(cfg.msname, readonly=True)
    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in set(t_main.colnames()) else 'TIME'
    it = t_main.iter([time_col], sort=True)
    total_chunks = sum(1 for _ in it)
    print(f"Found {total_chunks} time chunks in MS: {cfg.msname}")
    t_main.close()

    chunk_size = max(1, args.chunk_size)
    chunk_bounds = []
    start = 0
    while start < total_chunks:
        end = min(start + chunk_size - 1, total_chunks - 1)
        chunk_bounds.append((start, end))
        start = end + 1

    # Execute chunks: serial, Dask-Local, or Dask-SLURM
    agg_list = []
    if args.parallel_mode == 'serial':
        for (start, end) in chunk_bounds:
            print(f"[Serial] Chunk {start}..{end}")
            times, cube, c, m, M2 = process_chunk_task(cfg, ms_base, candidates_dir, start, end)
            agg_list.append((times, cube, c, m, M2))

    elif args.parallel_mode == 'dask-local':
        from dask.distributed import Client, LocalCluster
        n_workers = args.dask_workers if args.dask_workers > 0 else None
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=args.threads_per_worker,
            processes=(args.dask_scheduler == 'processes')
        )
        with Client(cluster) as client:
            futures = [client.submit(process_chunk_task, cfg, ms_base, candidates_dir, s, e)
                       for (s, e) in chunk_bounds]
            agg_list = client.gather(futures)
        cluster.close()

    elif args.parallel_mode == 'dask-slurm':
        # SLURMCluster (requires dask_jobqueue)
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster(
            queue=args.slurm_partition,
            account=args.slurm_account,
            cores=args.slurm_cores_per_worker,
            memory=args.slurm_mem,
            walltime=args.slurm_walltime,
            job_extra_directives=args.slurm_job_extra,
            interface=args.slurm_interface
        )
        # scale to number of workers (or adapt if dask_workers==0)
        if args.dask_workers and args.dask_workers > 0:
            cluster.scale(args.dask_workers)
        else:
            # adaptively allocate between 1 and len(chunk_bounds) workers
            cluster.adapt(minimum=1, maximum=max(1, len(chunk_bounds)))
        from dask.distributed import Client
        with Client(cluster) as client:
            futures = [client.submit(process_chunk_task, cfg, ms_base, candidates_dir, s, e)
                       for (s, e) in chunk_bounds]
            agg_list = client.gather(futures)
        cluster.close()

    else:
        raise ValueError(f"Unknown parallel_mode: {args.parallel_mode}")
    
    _ = finalise_welford_parallel(cfg, agg_list, run_final_variance=True)
    
    # Consolidate per-chunk catalogues into candidates/
    var_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}*_var_candidates.csv")
    box_pattern = os.path.join(cfg.candidates_dir, f"{cfg.ms_base}_chunk_*_boxcar_candidates.csv")
    candidates.consolidate_chunk_catalogues(
        ms_base=cfg.ms_base,
        out_dir=cfg.candidates_dir,
        var_csv_pattern=var_pattern,
        box_csv_pattern=box_pattern,
        remove_chunk_catalogues=True
    )
    
if __name__ == '__main__':
    main()
