#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 11: Figure 7 and the pattern statistics of Section 3.4 / SI Section S1.7.4 on a COMMON grid.
The vegetation masks (13.1 m x 13.2 m) and the contamination masks (5.00 m x 5.04 m) have different pixel sizes, so
box-counting fractal dimension and lacunarity are computed here on the vegetation grid, to which each contamination mask
is resampled by majority rule (area fraction >= 0.5). Both classes are then compared over the same metric scales:
box sizes 1 to 256 cells (13 m to 3.4 km), sub-ranges 1-8, 8-64 and 32-256 cells, gliding boxes 3 to 241 cells.
Screened scenes only (flag == False in the step 3/4 output). Writes fd_common_grid.csv, lacunarity_common_grid.csv,
fd_common_numbers.json and Figure_7_fractal_multiscale.png/tiff/eps."""
import argparse, json, os, sys
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter, NullLocator
from PIL import Image
from scipy import stats
import epsexport
Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--masks', required=True, help='folder with <n>_vegetation_mask.tiff and <n>_contamination_mask.tiff')
ap.add_argument('--monthly', required=True, help='step 3/4 output with the screening flag (monthly_final_km2.csv)')
ap.add_argument('--out', default='.'); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
H, W = config.COMPOSITE_SHAPE; Hv, Wv = config.VEGMASK_SHAPE
pE, pN = config.COMPOSITE_PX_EW_M, config.COMPOSITE_PX_NS_M; pEv, pNv = pE * W / Wv, pN * H / Hv
PX = 13.1   # nominal cell size used for the metric axis labels
print(f'common grid: vegetation grid {pEv:.2f} m x {pNv:.2f} m')
m = pd.read_csv(args.monthly, parse_dates=['date']).sort_values('date'); keep = m[~m.flag]

def load(n, kind):
    a = np.array(Image.open(os.path.join(args.masks, f'{n}_{kind}_mask.tiff'))) > 0
    tgt = (Wv, Hv) if kind == 'vegetation' else (W, H)
    if a.shape != (tgt[1], tgt[0]): a = np.array(Image.fromarray(a.astype(np.uint8) * 255).resize(tgt, Image.NEAREST)) > 0
    return a
def to_common(a):   # contamination mask -> vegetation grid by majority rule
    return cv2.resize(a.astype(np.float32), (Wv, Hv), interpolation=cv2.INTER_AREA) >= 0.5
def boxcount(mask, sizes):
    Hh, Ww = mask.shape; out = []
    for s in sizes:
        h = (Hh // s) * s; w = (Ww // s) * s
        out.append(int(mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3)).sum()))
    return np.array(out)
def fd_fit(sizes, counts):
    ok = counts > 0
    if ok.sum() < 3: return np.nan, np.nan
    x = np.log(1 / sizes[ok]); y = np.log(counts[ok]); p = np.polyfit(x, y, 1); yhat = np.polyval(p, x)
    return p[0], 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def lacunarity(mask, r, stride=None):
    Hh, Ww = mask.shape; stride = stride or max(1, r // 2)
    cs = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, Hh - r + 1, stride); xs = np.arange(0, Ww - r + 1, stride); Y, X = np.meshgrid(ys, xs, indexing='ij')
    mass = cs[Y + r, X + r] - cs[Y, X + r] - cs[Y + r, X] + cs[Y, X]; mu = mass.mean(); var = mass.var()
    return (var / mu ** 2 + 1) if mu > 0 else np.nan

sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]); lac_sizes = np.array([3, 5, 7, 11, 21, 31, 61, 121, 241])
fd_rows = []; lac_rows = []; area_rows = []
for n, d in zip(keep.image, keep.date):
    veg = load(int(n), 'vegetation'); con_full = load(int(n), 'contamination'); con = to_common(con_full)
    area_rows.append(dict(image=int(n), date=d, con_km2_native=con_full.sum() * pE * pN / 1e6, con_km2_common=con.sum() * pEv * pNv / 1e6, veg_km2=veg.sum() * pEv * pNv / 1e6))
    for cls, mk in (('vegetation', veg), ('contamination', con)):
        c = boxcount(mk, sizes); full, r2 = fd_fit(sizes, c); fine, _ = fd_fit(sizes[:4], c[:4]); mid, _ = fd_fit(sizes[3:7], c[3:7]); coarse, _ = fd_fit(sizes[5:], c[5:])
        fd_rows.append(dict(image=int(n), date=d, cls=cls, fd_full=full, r2_full=r2, fd_fine_1_8=fine, fd_mid_8_64=mid, fd_coarse_32_256=coarse, counts=list(map(int, c))))
        row = dict(image=int(n), date=d, cls=cls)
        for r in lac_sizes: row[f'L{r}'] = lacunarity(mk, int(r))
        lac_rows.append(row)
fd = pd.DataFrame(fd_rows); lac = pd.DataFrame(lac_rows); ar = pd.DataFrame(area_rows)
fd.to_csv(os.path.join(args.out, 'fd_common_grid.csv'), index=False); lac.to_csv(os.path.join(args.out, 'lacunarity_common_grid.csv'), index=False)
v = fd[fd.cls == 'vegetation']; c = fd[fd.cls == 'contamination']
t, p = stats.ttest_ind(v.fd_full, c.fd_full, equal_var=False); d = (c.fd_full.mean() - v.fd_full.mean()) / np.sqrt((v.fd_full.var(ddof=1) + c.fd_full.var(ddof=1)) / 2)
lv = lac[lac.cls == 'vegetation'][[f'L{r}' for r in lac_sizes]].median(); lc = lac[lac.cls == 'contamination'][[f'L{r}' for r in lac_sizes]].median()
lvs = lac[lac.cls == 'vegetation'].set_index('image')[[f'L{r}' for r in lac_sizes]]; lcs = lac[lac.cls == 'contamination'].set_index('image')[[f'L{r}' for r in lac_sizes]]
numbers = dict(n_scenes=int(len(v)), fd_veg=float(v.fd_full.mean()), fd_veg_sd=float(v.fd_full.std(ddof=1)), fd_con=float(c.fd_full.mean()), fd_con_sd=float(c.fd_full.std(ddof=1)), welch_t=float(t), p=float(p), cohen_d=float(d),
               fine_veg=float(v.fd_fine_1_8.mean()), fine_con=float(c.fd_fine_1_8.mean()), mid_veg=float(v.fd_mid_8_64.mean()), mid_con=float(c.fd_mid_8_64.mean()), coarse_veg=float(v.fd_coarse_32_256.mean()), coarse_con=float(c.fd_coarse_32_256.mean()),
               scenes_veg_below_con_fine=int((v.fd_fine_1_8.values < c.fd_fine_1_8.values).sum()), scenes_veg_below_con_mid=int((v.fd_mid_8_64.values < c.fd_mid_8_64.values).sum()), scenes_veg_below_con_coarse=int((v.fd_coarse_32_256.values < c.fd_coarse_32_256.values).sum()),
               lac_veg_median=lv.round(2).to_dict(), lac_con_median=lc.round(2).to_dict(), scenes_veg_lac_above_con_all_sizes=int((lvs.values > lcs.values).all(axis=1).sum()),
               con_area_native_mean_km2=float(ar.con_km2_native.mean()), con_area_common_mean_km2=float(ar.con_km2_common.mean()))
json.dump(numbers, open(os.path.join(args.out, 'fd_common_numbers.json'), 'w'), indent=1)
print(f"FD vegetation {numbers['fd_veg']:.2f} +/- {numbers['fd_veg_sd']:.2f}, contamination {numbers['fd_con']:.2f} +/- {numbers['fd_con_sd']:.2f}; Welch t = {t:.1f}, p = {p:.1e}, d = {d:.2f}")

# ---- Figure 7 (axes in metres) ----
OK = {'veg': '#009E73', 'con': '#D55E00'}; sizes_m = sizes * PX; lac_m = lac_sizes * PX
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
ax = axes[0]
for cls, col in (('vegetation', OK['veg']), ('contamination', OK['con'])):
    C = np.array([np.array(x) for x in fd[fd.cls == cls].counts]); C = np.where(C > 0, C, np.nan)
    ax.plot(sizes_m, np.nanmedian(C, axis=0), 'o-', color=col, ms=3, lw=1.2, label=cls); ax.fill_between(sizes_m, np.nanpercentile(C, 25, axis=0), np.nanpercentile(C, 75, axis=0), color=col, alpha=0.2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([13, 50, 200, 800, 3300])); ax.xaxis.set_major_formatter(FixedFormatter(['13', '50', '200', '800', '3,300'])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Box size (m)'); ax.set_ylabel('Occupied boxes N(ε)'); ax.legend(fontsize=7, frameon=False); ax.set_title('a) Box counting', fontsize=9, loc='left')
ax = axes[1]; labels = ['13–105 m', '105–840 m', '420–3,360 m', '13–3,360 m']; cols = ['fd_fine_1_8', 'fd_mid_8_64', 'fd_coarse_32_256', 'fd_full']
for kk, (cls, col) in enumerate((('vegetation', OK['veg']), ('contamination', OK['con']))):
    sub = fd[fd.cls == cls]; bp = ax.boxplot([sub[cc].dropna().values for cc in cols], positions=np.arange(4) + kk * 0.35 - 0.17, widths=0.3, patch_artist=True, showfliers=False, medianprops={'color': 'k', 'lw': 1})
    for b in bp['boxes']: b.set(facecolor=col, alpha=0.5, edgecolor=col)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor'); ax.set_xlabel('Box-size range'); ax.set_ylabel('Fractal dimension'); ax.set_title('b) FD by scale range', fontsize=9, loc='left')
ax = axes[2]
for cls, col in (('vegetation', OK['veg']), ('contamination', OK['con'])):
    sub = lac[lac.cls == cls][[f'L{r}' for r in lac_sizes]].values
    ax.plot(lac_m, np.nanmedian(sub, axis=0), 'o-', color=col, ms=3, lw=1.2, label=cls); ax.fill_between(lac_m, np.nanpercentile(sub, 25, axis=0), np.nanpercentile(sub, 75, axis=0), color=col, alpha=0.2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([40, 150, 600, 2500])); ax.xaxis.set_major_formatter(FixedFormatter(['40', '150', '600', '2,500'])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel('Gliding-box size (m)'); ax.set_ylabel('Lacunarity Λ'); ax.set_title('c) Lacunarity by scale', fontsize=9, loc='left')
fig.tight_layout(); base = os.path.join(args.out, 'Figure_7_fractal_multiscale')
fig.savefig(base + '.png', dpi=300, bbox_inches='tight'); fig.savefig(base + '.tiff', dpi=1000, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'}); epsexport.save_eps(fig, base + '.eps'); plt.close(fig)
print('Figure 7 written to', args.out)
