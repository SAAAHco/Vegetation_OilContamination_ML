#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 10: Figure S2, annual detection-frequency maps of the vegetation and contamination classes for the calendar years
2019 to 2024 from the screened monthly masks (Section S2.2). 2024 covers January and February only. Layers are composited to
RGB before drawing (PostScript has no transparency); the EPS uses block-averaged grids as Figure 6 does. Also writes
fig_s2_annual_stats.csv with the per-year areas at the 25 % frequency level, at least one detection, and the largest
8-connected contamination patch."""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from PIL import Image
from scipy import ndimage
import cv2, epsexport
Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--masks', required=True, help='folder with <n>_vegetation_mask.tiff and <n>_contamination_mask.tiff')
ap.add_argument('--composite', required=True, help='one composite tiff for the polygon outline')
ap.add_argument('--monthly', required=True, help='step 3 output with the screening flag (composite_radiometry.csv or monthly_final_km2.csv)')
ap.add_argument('--out', default='.'); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
pE, pN = config.COMPOSITE_PX_EW_M, config.COMPOSITE_PX_NS_M; PXC = config.KM2_PER_CONTAMINATION_PIXEL; PXV = config.KM2_PER_VEGETATION_PIXEL
H, W = config.COMPOSITE_SHAPE; Hv, Wv = config.VEGMASK_SHAPE; pEv, pNv = pE * W / Wv, pN * H / Hv   # vegetation-grid pixel size (m)
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
m = pd.read_csv(args.monthly, parse_dates=['date']).sort_values('date'); keep = m[~m.flag]
print('scenes failing the radiometric screen (excluded):', ', '.join(d.strftime('%Y-%m-%d') for d in m[m.flag].date))
print('screened scenes not used (outside 2019-2024):', ', '.join(d.strftime('%Y-%m-%d') for d in keep[~keep.date.dt.year.isin(YEARS)].date))

def load(n, kind):
    a = np.array(Image.open(os.path.join(args.masks, f'{n}_{kind}_mask.tiff'))) > 0
    tgt = (Wv, Hv) if kind == 'vegetation' else (W, H)
    if a.shape != (tgt[1], tgt[0]): a = np.array(Image.fromarray(a.astype(np.uint8) * 255).resize(tgt, Image.NEAREST)) > 0
    return a
def freq(rows, kind):
    f = None
    for n in rows.image:
        a = load(int(n), kind).astype('float32'); f = a if f is None else f + a
    return f / len(rows)
poly = np.array(Image.open(args.composite).convert('RGB')).sum(axis=2) > 30
poly_e = cv2.erode(poly.astype(np.uint8), np.ones((2 * config.POLYGON_EDGE_ERODE_PX + 1,) * 2, np.uint8)) > 0     # drop a 5-pixel (25 m) border where the composite edge is classed as contamination
poly_v = cv2.resize(poly_e.astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
poly_vfull = cv2.resize(poly.astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
years = {y: keep[keep.date.dt.year == y] for y in YEARS}
CF, VF = {}, {}
for y in YEARS:
    CF[y] = freq(years[y], 'contamination') * poly_e; VF[y] = freq(years[y], 'vegetation') * poly_v
    print(f'{y}: n = {len(years[y])} screened scenes ({", ".join(d.strftime("%b") for d in years[y].date)})')

# ---- statistics: areas at >= 25 % and >= once, largest 8-connected contamination patch at the 25 % level, vegetation cluster
S8 = np.ones((3, 3), bool)
def bbox_km(mask, px, py):
    ys, xs = np.nonzero(mask); return xs.min() * px / 1000, (xs.max() + 1) * px / 1000, ys.min() * py / 1000, (ys.max() + 1) * py / 1000
def largest_patch(mask):
    lab, n = ndimage.label(mask, structure=S8)
    if n == 0: return 0, np.zeros_like(mask), n, np.array([])
    sz = np.bincount(lab.ravel())[1:]; k = sz.argmax() + 1; return sz[k - 1], lab == k, n, sz
def main_cluster(vmask, weight, dil=5):
    """Largest group of vegetation pixels (8-connected after a 5-pixel = 65 m dilation so that neighbouring fields join)."""
    if not vmask.any(): return np.zeros_like(vmask)
    lab, n = ndimage.label(ndimage.binary_dilation(vmask, structure=S8, iterations=dil), structure=S8)
    tot = ndimage.sum(weight, lab, index=np.arange(1, n + 1)); return (lab == (np.argmax(tot) + 1)) & vmask
ntot = sum(len(years[y]) for y in YEARS); vf_all = sum(VF[y] * len(years[y]) for y in YEARS) / ntot
pooled = main_cluster(vf_all >= 0.10, vf_all); pb = bbox_km(pooled, pEv, pNv)
print(f'\nMain vegetation cluster (pooled 2019-2024 frequency >= 10 %, largest connected group): east {pb[0]:.1f}-{pb[1]:.1f} km, south {pb[2]:.1f}-{pb[3]:.1f} km, '
      f'{pooled.sum() * PXV:.3f} km2 = {100 * pooled.sum() / (vf_all >= 0.10).sum():.0f} % of the pooled >= 10 % area')
inbox = np.zeros_like(pooled); inbox[int(pb[2] * 1000 / pNv):int(np.ceil(pb[3] * 1000 / pNv)), int(pb[0] * 1000 / pEv):int(np.ceil(pb[1] * 1000 / pEv))] = True
c25_2019 = CF[2019] >= 0.25; _, patch2019, _, _ = largest_patch(c25_2019)
rows = []
for y in YEARS:
    cf, vf = CF[y], VF[y]; c25, v25 = cf >= 0.25, vf >= 0.25
    npx, pmask, ncomp, sz = largest_patch(c25); cb = bbox_km(pmask, pE, pN) if npx else (np.nan,) * 4
    vmain = main_cluster(v25, vf); vb = bbox_km(vmain, pEv, pNv) if vmain.any() else (np.nan,) * 4
    rows.append(dict(year=y, n_scenes=len(years[y]),
                     veg_ge25_km2=v25.sum() * PXV, veg_once_km2=(vf > 0).sum() * PXV, veg_ge25_in_main_box_pct=100 * (v25 & inbox).sum() / max(v25.sum(), 1),
                     veg_main_cluster_km2=vmain.sum() * PXV, veg_main_E0=vb[0], veg_main_E1=vb[1], veg_main_S0=vb[2], veg_main_S1=vb[3],
                     con_ge25_km2=c25.sum() * PXC, con_once_km2=(cf > 0).sum() * PXC, con_ge25_patches=ncomp, con_ge25_patches_gt0p05km2=int((sz * PXC > 0.05).sum()),
                     con_largest_patch_km2=npx * PXC, con_largest_E0=cb[0], con_largest_E1=cb[1], con_largest_S0=cb[2], con_largest_S1=cb[3],
                     con_2019ge25_still_ge25_pct=100 * (c25_2019 & c25).sum() / c25_2019.sum(), con_2019patch_ge25_pct=100 * (patch2019 & c25).sum() / patch2019.sum(),
                     con_2019patch_mean_freq_pct=100 * cf[patch2019].mean()))
st = pd.DataFrame(rows).set_index('year'); st.to_csv(os.path.join(args.out, 'fig_s2_annual_stats.csv'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\n', st.round(3).T.to_string())

# ---- figure: 2 rows (vegetation, contamination) x 6 years, one colorbar per row, km axes as in Figure 6
def ds(a, f, mode='mean'):
    if f == 1: return a.astype('float32') if mode == 'mean' else a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f; a = a[:h, :w]
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype='float32') if mode == 'mean' else a[::f, ::f]
def comp(base, vals, show, cmap, vmin, vmax):
    rgb = np.repeat(base[..., None], 3, axis=2).astype('float32'); sm = cm.ScalarMappable(mcolors.Normalize(vmin, vmax), cmap); sm.set_array([])
    col = sm.to_rgba(vals)[..., :3]; rgb[show] = col[show]; return rgb, sm
def make_fig(fc_, fv_):
    pc_ = ds(poly, fc_); pv_ = ds(poly_vfull, fv_); base_c = 1 - 0.08 * pc_; base_v = 1 - 0.08 * pv_
    ext_c = [0, pc_.shape[1] * fc_ * pE / 1000, pc_.shape[0] * fc_ * pN / 1000, 0]
    ext_v = [0, pv_.shape[1] * fv_ * pEv / 1000, pv_.shape[0] * fv_ * pNv / 1000, 0]
    fig, axes = plt.subplots(2, 6, figsize=(7.4, 2.7), sharex=True, sharey=True)   # panel size is width-limited; 2.7 in avoids blank rows
    for j, y in enumerate(YEARS):
        n = len(years[y]); ttl = f'{y} (n = {n})' if y != 2024 else f'{y} (Jan–Feb)\n(n = {n})'
        a = ds(VF[y], fv_); rgb, sm_v = comp(base_v, a * 100, a > 0, 'Greens', 0, 60); axes[0, j].imshow(rgb, extent=ext_v)
        a = ds(CF[y], fc_); rgb, sm_c = comp(base_c, a * 100, a > 0, 'Reds', 0, 100); axes[1, j].imshow(rgb, extent=ext_c)
        for ax in axes[:, j]: ax.set_title(ttl, fontsize=7.5)
    for ax in axes.ravel(): ax.set_aspect('equal'); ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10]); ax.tick_params(labelsize=7, length=2.5, pad=1.5)
    for ax in axes[1]: ax.set_xlabel('East (km)', fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel('South (km)', fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.955, 0.985], w_pad=0.6, h_pad=1.6)
    fig.canvas.draw(); rend = fig.canvas.get_renderer()      # positions after the equal-aspect boxes are applied
    for r, (sm, ticks, head) in enumerate(((sm_v, [0, 20, 40, 60], 'a) Vegetation detection frequency (%)'),
                                           (sm_c, [0, 25, 50, 75, 100], 'b) Contamination detection frequency (%)'))):
        p = axes[r, -1].get_position(); cax = fig.add_axes([p.x1 + 0.008, p.y0, 0.010, p.height])
        cb = fig.colorbar(sm, cax=cax, ticks=ticks); cb.ax.tick_params(labelsize=7, length=2.5, pad=1.5); cb.outline.set_linewidth(0.6)
        p0 = axes[r, 0].get_position(); tb = axes[r, 0].title.get_window_extent(rend).transformed(fig.transFigure.inverted())
        fig.text(p0.x0, tb.y1 + 0.012, head, fontsize=8, ha='left', va='bottom')
    return fig
fig = make_fig(1, 1); fig.savefig(os.path.join(args.out, 'Figure_S2_annual_maps.png'), dpi=300); fig.savefig(os.path.join(args.out, 'Figure_S2_annual_maps.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}); plt.close(fig)
fig = make_fig(4, 2); epsexport.save_eps(fig, os.path.join(args.out, 'Figure_S2_annual_maps.eps'), tight=False); plt.close(fig)
print('saved Figure_S2_annual_maps (png 300 dpi, tiff 600 dpi, eps block-averaged 4/2)')
