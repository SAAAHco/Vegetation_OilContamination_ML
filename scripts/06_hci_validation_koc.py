#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 6: validation of the Landsat HCI against the KOC scope-of-work contamination polygons digitized from the
Figure 1 map (Section S1.3, Figure S6). The map (UTM 38N, 1:40,000 A3) is georeferenced from its 1 km grid ticks:
easting = 780000 + (x - 185) / 133.6 * 1000 m, northing = 3203000 - (y - 180.4) / 133.6 * 1000 m for the 2244 x 1587
pixel image; adjust --e0/--n0/--ppk for another export of the map."""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
import planetary_computer as pc, pystac_client, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
from sklearn.metrics import roc_auc_score, roc_curve
import epsexport
Image.MAX_IMAGE_PIXELS = None

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', required=True, help='Figure 1 map image (KOC SKETR-II SOW map)')
    ap.add_argument('--e0', type=float, default=780000); ap.add_argument('--x0', type=float, default=185); ap.add_argument('--n0', type=float, default=3203000); ap.add_argument('--y0', type=float, default=180.4); ap.add_argument('--ppk', type=float, default=133.6, help='map pixels per km')
    ap.add_argument('--frame', default='56,2188,55,1290', help='map frame x0,x1,y0,y1 in pixels')
    ap.add_argument('--scenes', default='LC08_L2SP_165040_20191010_02_T1,LC09_L2SP_165040_20221010_02_T1,LC09_L2SP_165040_20231013_02_T1')
    ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    x0, x1, y0, y1 = [int(v) for v in a.frame.split(',')]
    arr = np.array(Image.open(a.map).convert('RGB')).astype(int); q = (arr[y0:y1, x0:x1] // 16) * 16; R, G, B = q[..., 0], q[..., 1], q[..., 2]
    cont = ((B >= 160) & (R >= 96) & (G <= 112)) | ((G >= 128) & (R <= 176) & (B <= 128) & (G > R + 30)) | ((R >= 96) & (G <= 64) & (B <= 64)) | ((R >= 192) & (B >= 208) & (G >= 144) & (G <= 208) & (B > G + 20))
    facil = ((R >= 232) & (G >= 232) & (B <= 224) & (B >= 200)) | ((R >= 232) & (G >= 200) & (G <= 224) & (B >= 200) & (B <= 224))
    px = 1000 / a.ppk; map_tr = from_origin(a.e0 + (x0 - a.x0) * px, a.n0 - (y0 - a.y0) * px, px, px)
    win_ll = transform_bounds('EPSG:32638', 'EPSG:4326', a.e0 + (x0 - a.x0) * px, a.n0 - (y1 - a.y0) * px, a.e0 + (x1 - a.x0) * px, a.n0 - (y0 - a.y0) * px)
    print(f'mapped contamination polygons: {cont.sum() * px * px / 1e6:.2f} km2 in the map frame')
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    ids = a.scenes.split(','); fig, axes = plt.subplots(2, len(ids), figsize=(2.4 * len(ids), 4.6)); rows = []
    for k, sid in enumerate(ids):
        it = list(cat.search(collections=['landsat-c2-l2'], ids=[sid]).items())[0]
        with rasterio.open(it.assets['red'].href) as src:
            win = from_bounds(*transform_bounds('EPSG:4326', src.crs, *win_ll), transform=src.transform); red = src.read(1, window=win).astype('float64'); wt = src.window_transform(win); crs = src.crs
        with rasterio.open(it.assets['swir22'].href) as src:
            sw = src.read(1, window=from_bounds(*transform_bounds('EPSG:4326', src.crs, *win_ll), transform=src.transform)).astype('float64')
        h = min(red.shape[0], sw.shape[0]); w = min(red.shape[1], sw.shape[1]); red, sw = red[:h, :w], sw[:h, :w]; r = red * 0.0000275 - 0.2; s = sw * 0.0000275 - 0.2; ok = (red > 0) & (sw > 0); hci = (s - r) / (s + r + 1e-9)
        def to_ls(mask):
            dst = np.zeros((h, w), 'float32'); reproject(mask.astype('float32'), dst, src_transform=map_tr, src_crs='EPSG:32638', dst_transform=wt, dst_crs=crs, resampling=Resampling.average); return dst
        cf = to_ls(cont); ff = to_ls(facil); inside = to_ls(np.ones_like(cont)) > 0.99; pos = (cf >= 0.5) & ok & inside; neg = (cf == 0) & (ff == 0) & ok & inside
        y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]; xh = np.r_[hci[pos], hci[neg]]; auc = roc_auc_score(y, xh); fpr, tpr, thr = roc_curve(y, xh)
        sens10 = float((hci[pos] > 0.10).mean()); spec10 = float((hci[neg] <= 0.10).mean())
        rows.append(dict(scene=sid, n_pos=int(pos.sum()), n_neg=int(neg.sum()), med_pos=float(np.median(hci[pos])), med_neg=float(np.median(hci[neg])), auc=float(auc), sens_at_010=sens10, spec_at_010=spec10))
        print(f'{sid}: AUC {auc:.2f}; median HCI contaminated {np.median(hci[pos]):.3f} clean {np.median(hci[neg]):.3f}; at 0.10 sensitivity {sens10:.2f} specificity {spec10:.2f}')
        title = f"{pd.Timestamp(sid[17:25]).strftime('%d %b %Y')} ({ {'LC08': 'Landsat 8', 'LC09': 'Landsat 9'}[sid[:4]] })"
        ax = axes[0, k]; bins = np.linspace(-0.05, 0.3, 50)
        ax.hist(hci[neg], bins=bins, density=True, alpha=0.5, color='#999999', label=f'Clean (n = {int(neg.sum()):,})')
        ax.hist(hci[pos], bins=bins, density=True, alpha=0.6, color='#D55E00', label=f'Contaminated (n = {int(pos.sum()):,})')
        ax.axvline(0.10, color='k', lw=0.7, ls='--'); ax.set_title(title, fontsize=8, loc='left'); ax.set_xlabel('HCI')
        if k == 0: ax.set_ylabel('Density'); ax.legend(fontsize=6, frameon=False)
        ax = axes[1, k]; ax.plot(fpr, tpr, color='#0072B2', lw=1.4); ax.plot([0, 1], [0, 1], 'k:', lw=0.8); ax.set_xlabel('False positive rate'); ax.set_title(f'AUC = {auc:.2f}', fontsize=8, loc='left')
        if k == 0: ax.set_ylabel('True positive rate')
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'hci_roc_vs_koc.csv'), index=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S6_hci_validation.png'), dpi=300); fig.savefig(os.path.join(a.out, 'Figure_S6_hci_validation.tiff'), dpi=1000, pil_kwargs={'compression': 'tiff_lzw'})
    epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S6_hci_validation.eps'), tight=False); plt.close(fig); print('saved to', a.out)

if __name__ == '__main__':
    main()
