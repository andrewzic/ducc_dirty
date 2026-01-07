# ducc_dirty

## installation
```
git clone https://github.com/andrewzic/fastducc.git
cd fastducc
pip install -e .
```
## usage
``` 
> fastducc -h
usage: fastducc [-h] --msname MSNAME [--chunk-size CHUNK_SIZE] [--corr-mode {average,stokesI,single}] [--basis {auto,linear,circular}] [--single-pol SINGLE_POL] [--data-column DATA_COLUMN] [--npix-x NPIX_X] [--npix-y NPIX_Y] [--pixsize-arcsec PIXSIZE_ARCSEC] [--epsilon EPSILON] [--do-wgridding] [--no-wgridding]
                [--nthreads NTHREADS] [--verbosity VERBOSITY] [--threshold-sigma THRESHOLD_SIGMA] [--do-plot] [--var-threshold-sigma VAR_THRESHOLD_SIGMA] [--var-keep-top-k VAR_KEEP_TOP_K] [--var-clip-sigma VAR_CLIP_SIGMA] [--var-nms-radius VAR_NMS_RADIUS]

Image MS in time chunks using ducc0 wgridder

options:
  -h, --help            show this help message and exit
  --msname MSNAME       Path to Measurement Set
  --chunk-size CHUNK_SIZE
                        Number of time samples per chunk (default: 1000)
  --corr-mode {average,stokesI,single}
                        Correlation handling mode (default: single)
  --basis {auto,linear,circular}
                        Basis for stokesI (default: auto)
  --single-pol SINGLE_POL
                        Single pol to image when corr-mode=single (default: XX)
  --data-column DATA_COLUMN
                        Which data column to image from the measurement set (default: DATA)
  --npix-x NPIX_X
  --npix-y NPIX_Y
  --pixsize-arcsec PIXSIZE_ARCSEC
                        Pixel size (arcsec); applied to both axes (default: 22.0)
  --epsilon EPSILON
  --do-wgridding
  --no-wgridding
  --nthreads NTHREADS
  --verbosity VERBOSITY
  --threshold-sigma THRESHOLD_SIGMA
                        S/N threshold to use for detections (default: 8.0)
  --do-plot
  --var-threshold-sigma VAR_THRESHOLD_SIGMA
  --var-keep-top-k VAR_KEEP_TOP_K
  --var-clip-sigma VAR_CLIP_SIGMA
  --var-nms-radius VAR_NMS_RADIUS
  ```
