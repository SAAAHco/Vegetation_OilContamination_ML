#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 5: per-pixel detection frequency and change classes (Figure 6), box-counting fractal dimension and lacunarity
across scales (Figure 7), and the vegetation-to-contamination distance profile (Figure S8) (Section 3.4).
Inputs: masks folder, one reference composite (for the polygon outline), and the monthly table from step 4 (image, date, flag)."""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from scipy import ndimage, stats
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config, epsexport
Image.MAX_IMAGE_PIXELS = None
H, W = config.COMPOSITE_SHAPE; Hv, Wv = config.VEGMASK_SHAPE

def load(masks, n, kind):
    a = np.array(Image.open(os.path.join(masks, f'{n}_{kind}_mask.tiff'))) > 0
    tgt = (Wv, Hv) if kind == 'vegetation' else (W, H)
    if a.shape != (tgt[1], tgt[0]): a = np.array(Image.fromarray(a.astype(np.uint8) * 255).resize(tgt, Image.NEAREST)) > 0
    return a

def boxcount(mask, sizes):
    Hm, Wm = mask.shape; out = []
    for s in sizes:
        h = (Hm // s) * s; w = (Wm // s) * s; out.append(int(mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3)).sum()))
    return np.array(out)

def fd_fit(sizes, counts):
    ok = counts > 0
    if ok.sum() < 3: return np.nan, np.nan
    x = np.log(1 / sizes[ok]); y = np.log(counts[ok]); p = np.polyfit(x, y, 1); yh = np.polyval(p, x)
    return p[0], 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def lacunarity(mask, r):
    Hm, Wm = mask.shape; stride = max(1, r // 2); cs = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, Hm - r + 1, stride); xs = np.arange(0, Wm - r + 1, stride); Y, X = np.meshgrid(ys, xs, indexing='ij')
    mass = cs[Y + r, X + r] - cs[Y, X + r] - cs[Y + r, X] + cs[Y, X]; mu = mass.mean(); return (mass.var() / mu ** 2 + 1) if mu > 0 else np.nan

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--masks', required=True); ap.add_argument('--composite', required=True, help='one composite tiff for the polygon outline')
    ap.add_argument('--monthly', required=True); ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    m = pd.read_csv(a.monthly, parse_dates=['date']); keep = m[~m.flag].sort_values('date')
    PXC = config.KM2_PER_CONTAMINATION_PIXEL; PXV = config.KM2_PER_VEGETATION_PIXEL
    poly = np.array(Image.open(a.composite).convert('RGB')).sum(axis=2) > 30
    if poly.shape != (H, W): poly = np.array(Image.fromarray(poly.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    k = 2 * config.POLYGON_EDGE_ERODE_PX + 1; poly_e = cv2.erode(poly.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    poly_v = cv2.resize(poly_e.astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
    pre = keep[(keep.date >= '2019-01-01') & (keep.date < config.REVEGETATION_START)]; post = keep[keep.date >= config.REVEGETATION_START]
    def freq(rows, kind):
        f = None
        for n in rows.image: x = load(a.masks, int(n), kind).astype('float32'); f = x if f is None else f + x
        return f / len(rows)
    cf_pre, cf_post = freq(pre, 'contamination') * poly_e, freq(post, 'contamination') * poly_e; vf_pre, vf_post = freq(pre, 'vegetation') * poly_v, freq(post, 'vegetation') * poly_v
    cf_all = (cf_pre * len(pre) + cf_post * len(post)) / (len(pre) + len(post)); vf_all = (vf_pre * len(pre) + vf_post * len(post)) / (len(pre) + len(post))
    P, R = config.PERSISTENT_MIN_FREQ, config.REMOVED_MAX_FREQ
    persist = (cf_pre >= P) & (cf_post >= P); removed = (cf_pre >= P) & (cf_post < R); new = (cf_pre < R) & (cf_post >= P)
    st = dict(n_pre=len(pre), n_post=len(post), con_ever_km2=((cf_pre > 0) | (cf_post > 0)).sum() * PXC, con_ge25_km2=(cf_all >= .25).sum() * PXC, con_ge75_km2=(cf_all >= .75).sum() * PXC,
              con_pre_ge50_km2=(cf_pre >= P).sum() * PXC, con_post_ge50_km2=(cf_post >= P).sum() * PXC, persistent_km2=persist.sum() * PXC, removed_km2=removed.sum() * PXC, new_km2=new.sum() * PXC,
              veg_ever_km2=((vf_pre > 0) | (vf_post > 0)).sum() * PXV, veg_ge25_km2=(vf_all >= .25).sum() * PXV, veg_ge50_km2=(vf_all >= .5).sum() * PXV,
              veg_pre_ge25_km2=(vf_pre >= .25).sum() * PXV, veg_post_ge25_km2=(vf_post >= .25).sum() * PXV, veg_gain_km2=((vf_pre < .1) & (vf_post >= .25)).sum() * PXV, veg_loss_km2=((vf_pre >= .25) & (vf_post < .1)).sum() * PXV)
    pd.Series(st).to_csv(os.path.join(a.out, 'spatial_change_stats.csv')); print(pd.Series(st).round(3).to_string())
    # Figure 6: each panel is composited to one RGB array so that PNG, TIFF and EPS render identically (PostScript has no transparency);
    # the EPS uses block-averaged grids (factor f on the composite grid, the matching factor on the coarser vegetation grid)
    poly_vfull = cv2.resize(poly.astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
    cls = np.zeros((H, W), 'uint8'); cls[persist] = 1; cls[removed] = 2; cls[new] = 3
    pvE, pvN = config.COMPOSITE_PX_EW_M * W / Wv, config.COMPOSITE_PX_NS_M * H / Hv
    def ds(x, f, mode='mean'):
        if f == 1: return x.astype('float32') if mode == 'mean' else x
        h, w = (x.shape[0] // f) * f, (x.shape[1] // f) * f; x = x[:h, :w]
        return x.reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype='float32') if mode == 'mean' else x[::f, ::f]
    def comp(base, vals, show, cmap, vmin, vmax):
        rgb = np.repeat(base[..., None], 3, axis=2).astype('float32'); sm = cm.ScalarMappable(mcolors.Normalize(vmin, vmax), cmap); sm.set_array([])
        col = sm.to_rgba(vals)[..., :3]; rgb[show] = col[show]; return rgb, sm
    def make_fig6(f):
        fv = max(1, round(f * Hv / H)); pc = ds(poly, f); pv = ds(poly_vfull, fv); base_c = 1 - 0.08 * pc; base_v = 1 - 0.08 * pv
        ext_c = [0, pc.shape[1] * f * config.COMPOSITE_PX_EW_M / 1000, pc.shape[0] * f * config.COMPOSITE_PX_NS_M / 1000, 0]; ext_v = [0, pv.shape[1] * fv * pvE / 1000, pv.shape[0] * fv * pvN / 1000, 0]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        x = ds(cf_all, f); rgb, sm = comp(base_c, x * 100, x > 0, 'Reds', 0, 100); ax = axes[0, 0]; ax.imshow(rgb, extent=ext_c); ax.set_title('a) Contamination detection frequency (%)', fontsize=8.5, loc='left'); plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        x = ds(vf_all, fv); rgb, sm = comp(base_v, x * 100, x > 0, 'Greens', 0, 60); ax = axes[0, 1]; ax.imshow(rgb, extent=ext_v); ax.set_title('b) Vegetation detection frequency (%)', fontsize=8.5, loc='left'); plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        c = ds(cls, f, 'nearest'); rgb = np.repeat(base_c[..., None], 3, axis=2).astype('float32')
        for v, col in ((1, '#7f7f7f'), (2, '#0072B2'), (3, '#D55E00')): rgb[c == v] = mcolors.to_rgb(col)
        ax = axes[1, 0]; ax.imshow(rgb, extent=ext_c, interpolation='nearest'); ax.set_title('c) Contamination change classes', fontsize=8.5, loc='left')
        ax.legend(handles=[Patch(color='#7f7f7f', label='persistent'), Patch(color='#0072B2', label='removed'), Patch(color='#D55E00', label='new')], fontsize=7, frameon=False, loc='lower left')
        d = ds(vf_post, fv) - ds(vf_pre, fv); empty = ds((vf_pre == 0) & (vf_post == 0), fv) >= 1; rgb, sm = comp(base_v, d * 100, ~empty, 'RdYlGn', -40, 40)
        ax = axes[1, 1]; ax.imshow(rgb, extent=ext_v); ax.set_title('d) Vegetation frequency change (pp)', fontsize=8.5, loc='left'); plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        for ax in axes.ravel(): ax.set_aspect('equal'); ax.set_xlabel('East (km)'); ax.set_ylabel('South (km)')
        fig.tight_layout(); return fig
    fig = make_fig6(1); fig.savefig(os.path.join(a.out, 'Figure_6_spatial_change.png'), dpi=300); fig.savefig(os.path.join(a.out, 'Figure_6_spatial_change.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}); plt.close(fig)
    fig = make_fig6(4); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_6_spatial_change.eps'), tight=False); plt.close(fig)
    # fractal dimension and lacunarity (all scenes)
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]); lac_sizes = [3, 5, 7, 11, 21, 31, 61, 121, 241]; fd_rows = []; lac_rows = []
    for n in keep.image:   # pattern metrics on the screened scenes only, as in the paper
        for cls_name, mask in (('vegetation', load(a.masks, int(n), 'vegetation')), ('contamination', load(a.masks, int(n), 'contamination'))):
            c = boxcount(mask, sizes); full, r2 = fd_fit(sizes, c); fine, _ = fd_fit(sizes[:4], c[:4]); mid, _ = fd_fit(sizes[3:7], c[3:7]); coarse, _ = fd_fit(sizes[5:], c[5:])
            fd_rows.append(dict(image=n, cls=cls_name, fd_full=full, r2_full=r2, fd_fine_1_8=fine, fd_mid_8_64=mid, fd_coarse_32_256=coarse, counts=[int(v) for v in c]))
            lac_rows.append(dict(image=n, cls=cls_name, **{f'L{r}': lacunarity(mask, r) for r in lac_sizes}))
    fd = pd.DataFrame(fd_rows); lac = pd.DataFrame(lac_rows); fd.to_csv(os.path.join(a.out, 'fd_multiscale.csv'), index=False); lac.to_csv(os.path.join(a.out, 'lacunarity_multiscale.csv'), index=False)
    v = fd[fd.cls == 'vegetation'].fd_full; c_ = fd[fd.cls == 'contamination'].fd_full; tst = stats.ttest_ind(v, c_, equal_var=False)
    print(f'FD vegetation {v.mean():.2f}±{v.std():.2f}; contamination {c_.mean():.2f}±{c_.std():.2f}; Welch t={tst.statistic:.1f} p={tst.pvalue:.2g}')
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9)); pairs = (('vegetation', '#009E73'), ('contamination', '#D55E00'))
    ax = axes[0]
    for cls_name, col in pairs:
        C = np.array(fd[fd.cls == cls_name].counts.tolist(), 'float'); C = np.where(C > 0, C, np.nan)
        ax.plot(sizes, np.nanmedian(C, axis=0), 'o-', color=col, ms=3, lw=1.2, label=cls_name); ax.fill_between(sizes, np.nanpercentile(C, 25, axis=0), np.nanpercentile(C, 75, axis=0), color=col, alpha=0.2)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256])); ax.xaxis.set_major_formatter(FixedFormatter(['1', '4', '16', '64', '256'])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Box size (pixels)'); ax.set_ylabel('Occupied boxes N(ε)'); ax.legend(fontsize=7, frameon=False); ax.set_title('a) Box counting', fontsize=9, loc='left')
    ax = axes[1]; cols = ['fd_fine_1_8', 'fd_mid_8_64', 'fd_coarse_32_256', 'fd_full']
    for kk, (cls_name, col) in enumerate(pairs):
        sub = fd[fd.cls == cls_name]; bp = ax.boxplot([sub[k].values for k in cols], positions=np.arange(4) + kk * 0.35 - 0.17, widths=0.3, patch_artist=True, showfliers=False, medianprops={'color': 'k', 'lw': 1})
        for b in bp['boxes']: b.set(facecolor=col, alpha=0.5, edgecolor=col)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(['1–8', '8–64', '32–256', '1–256'], fontsize=8, rotation=45, ha='right', rotation_mode='anchor'); ax.set_xlabel('Box-size range (pixels)'); ax.set_ylabel('Fractal dimension'); ax.set_title('b) FD by scale range', fontsize=9, loc='left')
    ax = axes[2]
    for cls_name, col in pairs:
        ls = lac[lac.cls == cls_name][[f'L{r}' for r in lac_sizes]].values
        ax.plot(lac_sizes, np.nanmedian(ls, axis=0), 'o-', color=col, ms=3, lw=1.2, label=cls_name); ax.fill_between(lac_sizes, np.nanpercentile(ls, 25, axis=0), np.nanpercentile(ls, 75, axis=0), color=col, alpha=0.2)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 240])); ax.xaxis.set_major_formatter(FixedFormatter(['3', '10', '30', '100', '240'])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Gliding-box size (pixels)'); ax.set_ylabel('Lacunarity Λ'); ax.set_title('c) Lacunarity by scale', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_7_fractal_multiscale.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(a.out, 'Figure_7_fractal_multiscale.tiff'), dpi=1000, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    epsexport.save_eps(fig, os.path.join(a.out, 'Figure_7_fractal_multiscale.eps'), tight=True); plt.close(fig)
    # distance profile (screened scenes)
    bins = np.arange(0, 3001, 100); prof = []
    for n in keep.image:
        con = load(a.masks, int(n), 'contamination'); veg = np.array(Image.fromarray(load(a.masks, int(n), 'vegetation').astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
        if con.sum() == 0: continue
        dist = ndimage.distance_transform_edt(~con, sampling=(config.COMPOSITE_PX_NS_M, config.COMPOSITE_PX_EW_M)); idx = np.digitize(dist, bins) - 1; row = []
        for b in range(len(bins) - 1):
            sel = (idx == b) & poly & ~con; row.append(veg[sel].mean() if sel.sum() > 1000 else np.nan)
        prof.append(row)
    prof = np.array(prof); g = pd.DataFrame({'dist_lo_m': bins[:-1], 'dist_hi_m': bins[1:], 'veg_frac_median': np.nanmedian(prof, axis=0), 'q1': np.nanpercentile(prof, 25, axis=0), 'q3': np.nanpercentile(prof, 75, axis=0)}); g.to_csv(os.path.join(a.out, 'gradient_profile.csv'), index=False)
    gg = g.dropna(); mid = (gg.dist_lo_m + gg.dist_hi_m) / 2; fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.fill_between(mid, gg.q1 * 100, gg.q3 * 100, color='#009E73', alpha=0.25, label='Interquartile range across scenes'); ax.plot(mid, gg.veg_frac_median * 100, 'o-', color='#009E73', ms=3, label='Median across scenes')
    ax.set_xlabel('Distance from nearest contamination pixel (m)'); ax.set_ylabel('Vegetation density (% of area)'); ax.set_xlim(0, 1000); ax.set_yscale('symlog', linthresh=0.01); ax.set_ylim(0, 1.5); ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S8_gradient.png'), dpi=300); fig.savefig(os.path.join(a.out, 'Figure_S8_gradient.tiff'), dpi=1000, pil_kwargs={'compression': 'tiff_lzw'}); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S8_gradient.eps'), tight=False); plt.close(fig)
    print('saved to', a.out)

if __name__ == '__main__':
    main()
