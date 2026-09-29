#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 12: check of the export-based vegetation class area against Level-2A surface reflectance (Section 2.4,
SI Section S2.8, Figure S7). For a sample of screened Sentinel-2 scenes the area of the analysis polygon in which
SAVI (L = 0.5) computed from the Level-2A bands B02, B04 and B08 (10 m) exceeds the vegetation-mask threshold (0.15)
is compared with the vegetation class area of the exported composite for the same acquisition. Bands are read from
the Copernicus Sentinel-2 Level-2A collection of the Microsoft Planetary Computer (network); the digital-number offset
of processing baselines 04.00 and later is removed; SCL classes 0, 1, 3, 8, 9, 10 and 11 are excluded. Only tiles in
UTM zone 38N are used so that no reprojection is needed. Writes l2a_check.csv and Figure_S7_l2a_check.png/tiff/eps."""
import argparse, os, sys
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
import epsexport
Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
DEFAULT_DATES = ['2019-03-19', '2019-04-21', '2019-06-25', '2019-10-10', '2019-12-29', '2020-02-27', '2020-05-30', '2020-08-28', '2021-02-19', '2021-08-30', '2021-11-28',
                 '2022-03-26', '2022-08-05', '2022-10-22', '2023-02-24', '2023-08-18', '2023-10-27', '2023-11-11', '2024-01-30', '2024-02-14']
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--composite', required=True, help='one composite tiff for the polygon outline')
ap.add_argument('--monthly', required=True, help='step 4 output (monthly_final_km2.csv)')
ap.add_argument('--dates', default=','.join(DEFAULT_DATES), help='comma-separated acquisition dates (screened Sentinel-2 scenes)')
ap.add_argument('--out', default='.'); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
import planetary_computer as pc, pystac_client, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
E0, N0 = config.COMPOSITE_ORIGIN_UTM38; dE, dN = config.COMPOSITE_PX_EW_M, config.COMPOSITE_PX_NS_M; H, W = config.COMPOSITE_SHAPE
win_utm = (E0, N0 - H * dN, E0 + W * dE, N0); win_ll = transform_bounds('EPSG:32638', 'EPSG:4326', *win_utm)
poly = np.array(Image.open(args.composite).convert('RGB')).sum(axis=2) > 30
poly_e = cv2.erode(poly.astype(np.uint8), np.ones((2 * config.POLYGON_EDGE_ERODE_PX + 1,) * 2, np.uint8)) > 0
m = pd.read_csv(args.monthly, parse_dates=['date']).sort_values('date')
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
rows = []
for ds in args.dates.split(','):
    r = m[m.date == pd.Timestamp(ds)]
    if len(r) == 0 or bool(r.flag.iloc[0]): print(ds, 'not a screened scene, skipped'); continue
    items = [i for i in cat.search(collections=['sentinel-2-l2a'], bbox=list(win_ll), datetime=f'{ds}/{ds}').items() if i.id.split('_')[4] in ('T38RQS', 'T38RQT')]
    if not items: print(ds, 'no zone-38 item'); continue
    it = sorted(items, key=lambda i: i.properties.get('eo:cloud_cover', 99))[0]; bands = {}
    for b in ['B02', 'B04', 'B08', 'SCL']:
        with rasterio.open(it.assets[b].href) as src:
            w = from_bounds(*transform_bounds('EPSG:32638', src.crs, *win_utm), transform=src.transform)
            bands[b] = src.read(1, window=w, out_shape=(int(round((win_utm[3] - win_utm[1]) / 10)), int(round((win_utm[2] - win_utm[0]) / 10))), resampling=rasterio.enums.Resampling.nearest)
    blue, red, nir = (bands[b].astype('float64') / 10000 for b in ('B02', 'B04', 'B08'))
    pb = it.properties.get('s2:processing_baseline', '00.00')
    if float(pb) >= 4.0: blue -= 0.1; red -= 0.1; nir -= 0.1
    L = config.SAVI_L; savi = (nir - red) / (nir + red + L) * (1 + L)
    good = ~np.isin(bands['SCL'], [0, 1, 3, 8, 9, 10, 11]) & (nir > 0) & (red > 0)
    hh, ww = savi.shape; Ey = win_utm[0] + (np.arange(ww) + 0.5) * 10; Ny = win_utm[3] - (np.arange(hh) + 0.5) * 10
    col = np.clip(((Ey - E0) / dE).astype(int), 0, W - 1); row = np.clip(((N0 - Ny) / dN).astype(int), 0, H - 1)
    inside = poly_e[row][:, col] & good
    thr = config.VEGETATION_SAVI_THRESHOLD
    rows.append(dict(date=ds, item=it.id, baseline=pb, cloud=it.properties.get('eo:cloud_cover'), n_inside=int(inside.sum()),
                     savi_median=float(np.median(savi[inside])), area_savi015_km2=float((savi[inside] > thr).sum() * 1e-4), area_savi010_km2=float((savi[inside] > 0.10).sum() * 1e-4),
                     area_savi020_km2=float((savi[inside] > 0.20).sum() * 1e-4), export_veg_km2=float(r.veg_km2.iloc[0]), export_con_km2=float(r.con_km2.iloc[0])))
    print(ds, it.id, f'SAVI > {thr}: {rows[-1]["area_savi015_km2"]:.3f} km2 | export vegetation class {rows[-1]["export_veg_km2"]:.3f} km2', flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, 'l2a_check.csv'), index=False)
rho, p = stats.spearmanr(df.area_savi015_km2, df.export_veg_km2); print(f'Spearman rho = {rho:.2f}, p = {p:.3f}, n = {len(df)}')
seas = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
cols = {'Winter': '#0072B2', 'Spring': '#009E73', 'Summer': '#D55E00', 'Autumn': '#CC79A7'}; df['season'] = [seas[pd.Timestamp(d).month] for d in df.date]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
for sname, col in cols.items():
    sub = df[df.season == sname]
    if len(sub): ax.scatter(sub.export_veg_km2, sub.area_savi015_km2, s=28, color=col, edgecolor='k', linewidth=0.4, label=sname, zorder=3)
ax.plot([0.05, 20], [0.05, 20], color='#777777', lw=0.8, ls='--', label='1:1'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 3); ax.set_ylim(0.1, 20)
for _, rw in df.sort_values('area_savi015_km2', ascending=False).head(3).iterrows(): ax.annotate(pd.Timestamp(rw.date).strftime('%b %Y'), (rw.export_veg_km2, rw.area_savi015_km2), xytext=(6, -3), textcoords='offset points', fontsize=7)
ax.set_xlabel('Vegetation class area from exports (km²)'); ax.set_ylabel('SAVI > 0.15 area from Level-2A (km²)'); ax.legend(fontsize=7, frameon=False, loc='lower right', ncol=2)
ax.text(0.03, 0.97, f'Spearman ρ = {rho:.2f}, p = {p:.2f}, n = {len(df)}', transform=ax.transAxes, va='top', fontsize=8)
fig.tight_layout(); base = os.path.join(args.out, 'Figure_S7_l2a_check')
fig.savefig(base + '.png', dpi=300); fig.savefig(base + '.tiff', dpi=1000, pil_kwargs={'compression': 'tiff_lzw'}); epsexport.save_eps(fig, base + '.eps'); plt.close(fig)
print('Figure S7 written to', args.out)
