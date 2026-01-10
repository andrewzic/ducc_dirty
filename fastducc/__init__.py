# fastducc/__init__.py


"""
fastducc: tools for imaging and transient detection with ducc0 and casacore.

Submodules:
- fastducc_run: CLI / orchestration to image MS in time chunks and run detection
- imaging: per-chunk dirty image generation (ducc0 wgridder)
- detection: boxcar/SNR maps
- kernels: numba-accelerated primitives (moving sums, MAD/RMS, temporal SNR, max-filter)
- filters: candidate filtering utils including  NMS, cross-width grouping
- ms_utils: Measurement Set helpers (corr labels, spectral window, times, field centers)
- wcs: sky projection/WCS utilities
- candidates: snippet extraction & packaging of candidate cutouts
- ducc_continuum_img: auxiliary continuum imaging tool
- types: type definitions for dataclasses
"""

__all__ = [
    "fastducc_run",
    "imaging",
    "detection",
    "kernels",
    "filters",
    "ms_utils",
    "wcs",
    "candidates",
    "types"
]

__version__ = "0.1.0"
