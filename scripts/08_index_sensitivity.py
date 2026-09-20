#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 8: sensitivity of SAVI to the soil-adjustment factor L, and NDVI and EVI statistics of bare soil, computed from
Sentinel-2 Level-2A blue, red and near-infrared bands over the analysis window (Section S1.3)."""
import argparse, os, sys
import numpy as np, planetary_computer as pc, pystac_client, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dates', default='2023-02-24,2023-10-27'); ap.add_argument('--threshold', type=float, default=config.VEGETATION_SAVI_THRESHOLD)
    ap.add_argument('--offset', type=float, default=-0.1, help='L2A BOA offset (processing baseline 04.00 and later: -0.1)'); a = ap.parse_args()
    win = list(config.WINDOW_LONLAT); res = 0.0001; W = int(round((win[2] - win[0]) / res)); H = int(round((win[3] - win[1]) / res)); tr = from_origin(win[0], win[3], res, res)
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace); rng = np.random.default_rng(0)
    for date in a.dates.split(','):
        items = list(cat.search(collections=['sentinel-2-l2a'], bbox=win, datetime=f'{date}/{date}').items()); items.sort(key=lambda i: i.properties.get('eo:cloud_cover', 99)); it = items[0]; b = {}
        for band in ['B02', 'B04', 'B08']:
            with rasterio.open(it.assets[band].href) as src:
                sb = transform_bounds('EPSG:4326', src.crs, *win); w = from_bounds(*sb, transform=src.transform); arr = src.read(1, window=w).astype('float32'); wt = src.window_transform(w)
                dst = np.zeros((H, W), 'float32'); reproject(arr, dst, src_transform=wt, src_crs=src.crs, dst_transform=tr, dst_crs='EPSG:4326', resampling=Resampling.bilinear); b[band] = dst * 0.0001 + a.offset
        blue, red, nir = b['B02'], b['B04'], b['B08']; ok = (red > 0) & (nir > 0); ndvi = (nir - red) / (nir + red); evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        savi = {L: (nir - red) / (nir + red + L) * (1 + L) for L in (0.25, 0.5, 0.75, 1.0)}
        print(f'=== {date} {it.id} cloud {it.properties.get("eo:cloud_cover"):.2f}')
        print(f'  NDVI median {np.median(ndvi[ok]):.3f}; EVI median {np.median(evi[ok]):.3f}; ' + '; '.join(f'SAVI(L={L}) median {np.median(savi[L][ok]):.3f}' for L in savi))
        idx = rng.choice(np.where(ok.ravel())[0], 200000, replace=False)
        for L in (0.25, 0.75, 1.0): print(f'  Spearman SAVI(0.5) vs SAVI({L}): {stats.spearmanr(savi[0.5].ravel()[idx], savi[L].ravel()[idx]).statistic:.4f}')
        print(f'  Spearman SAVI(0.5) vs NDVI {stats.spearmanr(savi[0.5].ravel()[idx], ndvi.ravel()[idx]).statistic:.3f}; vs EVI {stats.spearmanr(savi[0.5].ravel()[idx], evi.ravel()[idx]).statistic:.3f}')
        f05 = (savi[0.5][ok] > a.threshold).mean(); qq = 1 - f05
        for L in (0.25, 0.75, 1.0):
            fixed = (savi[L][ok] > a.threshold).mean(); print(f'  L={L}: vegetated fraction with fixed threshold {fixed:.4f} ({100 * (fixed / f05 - 1):+.0f}% vs L=0.5); quantile-equivalent threshold {np.quantile(savi[L][ok], qq):.3f}')
        bare = savi[0.5][ok] < np.median(savi[0.5][ok]); print(f'  bare soil (lower half) NDVI median {np.median(ndvi[ok][bare]):.3f}, SAVI(0.5) median {np.median(savi[0.5][ok][bare]):.3f}')

if __name__ == '__main__':
    main()
