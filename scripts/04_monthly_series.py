#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 4: monthly class areas in km2, trend tests, periodogram, seasonal and pre/post comparisons, climate correlations,
site-state sensitivity, and Figures 4, 5, S1, S3, S4, S5 with Tables 5, 6 and S3 (Sections 3.2 and 3.3).
Inputs: masks folder with <n>_vegetation_mask.tiff and <n>_contamination_mask.tiff, the radiometry table from step 3
(image, date, flag), and the monthly climate table from step 9 (index month, columns rain_mm, TAVG, TMAX)."""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter
from scipy import stats
from scipy.signal import lombscargle
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config, epsexport
Image.MAX_IMAGE_PIXELS = None
G, O, B, K = '#009E73', '#D55E00', '#0072B2', '#444444'
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']
SEASON_OF = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

def mann_kendall(x):
    n = len(x); s = sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18; z = (s - np.sign(s)) / np.sqrt(var); return z, 2 * (1 - stats.norm.cdf(abs(z)))

def ls_power(t, x, periods):
    x = (x - x.mean()) / x.std(); return lombscargle(t, x, 2 * np.pi / periods, normalize=True)

def ar1_levels(t, x, periods, nrep=2000, seed=1):
    rng = np.random.default_rng(seed); x = (x - x.mean()) / x.std(); r1 = max(np.corrcoef(x[:-1], x[1:])[0, 1], 0.0); sims = []
    for _ in range(nrep):
        e = rng.standard_normal(len(t)); s = np.zeros(len(t))
        for k in range(1, len(t)): s[k] = r1 * s[k - 1] + e[k] * np.sqrt(1 - r1 ** 2)
        sims.append(ls_power(t, s, periods))
    return r1, np.percentile(np.array(sims), [95, 99], axis=0)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--masks', required=True); ap.add_argument('--radiometry', required=True); ap.add_argument('--climate', required=True)
    ap.add_argument('--n', type=int, default=config.N_SCENES); ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    rad = pd.read_csv(a.radiometry, parse_dates=['date'])
    rows = []
    for n in range(1, a.n + 1):
        v = np.array(Image.open(os.path.join(a.masks, f'{n}_vegetation_mask.tiff'))) > 0; c = np.array(Image.open(os.path.join(a.masks, f'{n}_contamination_mask.tiff'))) > 0
        rows.append(dict(image=n, veg_px=int(v.sum()), con_px=int(c.sum())))
    m = pd.DataFrame(rows).merge(rad[['image', 'date', 'z_RminusB', 'flag']], on='image').sort_values('date').reset_index(drop=True)
    m['veg_km2'] = m.veg_px * config.KM2_PER_VEGETATION_PIXEL; m['con_km2'] = m.con_px * config.KM2_PER_CONTAMINATION_PIXEL
    m['season'] = m.date.dt.month.map(SEASON_OF); m['year'] = m.date.dt.year; m['ym'] = m.date.dt.to_period('M')
    c = pd.read_csv(a.climate, index_col=0, parse_dates=True); c['ym'] = c.index.to_period('M'); c = c.sort_index()
    c['rain_3mo'] = c.rain_mm.rolling(3).sum(); c['rain_6mo'] = c.rain_mm.rolling(6).sum()
    m = m.merge(c[['ym', 'rain_mm', 'rain_3mo', 'rain_6mo', 'TAVG', 'TMAX']], on='ym', how='left'); m.to_csv(os.path.join(a.out, 'monthly_final_km2.csv'), index=False)
    k = m[~m.flag].copy(); t = (k.date - m.date.min()).dt.days.values / 365.25; out = []
    res = {}
    for v in ['veg_km2', 'con_km2']:
        ts = stats.theilslopes(k[v].values, t); z, p = mann_kendall(k[v].values); res[v] = dict(slope=ts[0], intercept=ts[1], lo=ts[2], hi=ts[3], p=p)
        out.append(f'{v}: n={len(k)} mean {k[v].mean():.3f} sd {k[v].std():.3f} median {k[v].median():.3f}; Theil-Sen {ts[0]:+.4f} km2/yr (95% CI {ts[2]:+.4f} to {ts[3]:+.4f}); Mann-Kendall z={z:.2f} p={p:.3f}')
    seas = k.groupby('season')[['veg_km2', 'con_km2']].agg(['count', 'mean', 'std', 'median']).reindex(SEASONS); seas.to_csv(os.path.join(a.out, 'table5_seasonal.csv'))
    yr = k.groupby('year')[['veg_km2', 'con_km2']].agg(['count', 'mean', 'std', 'median']); yr.to_csv(os.path.join(a.out, 'table5_yearly.csv'))
    kw_v = stats.kruskal(*[g.veg_km2.values for _, g in k.groupby('season')]); kw_c = stats.kruskal(*[g.con_km2.values for _, g in k.groupby('season')])
    out.append(f'Kruskal-Wallis seasons: vegetation H={kw_v.statistic:.2f} p={kw_v.pvalue:.4f}; contamination H={kw_c.statistic:.2f} p={kw_c.pvalue:.4f}')
    pre = k[(k.date >= '2019-01-01') & (k.date < config.REVEGETATION_START)]; post = k[k.date >= config.REVEGETATION_START]
    for v in ['veg_km2', 'con_km2']:
        u = stats.mannwhitneyu(pre[v], post[v]); r = 1 - 2 * u.statistic / (len(pre) * len(post))
        out.append(f'pre/post {v}: pre n={len(pre)} median {pre[v].median():.3f} | post n={len(post)} median {post[v].median():.3f} | U={u.statistic:.0f} p={u.pvalue:.3f} r={r:.3f}')
    for v in ['rain_mm', 'rain_3mo', 'rain_6mo', 'TAVG', 'TMAX']:
        r1 = stats.spearmanr(k.veg_km2, k[v], nan_policy='omit'); r2 = stats.spearmanr(k.con_km2, k[v], nan_policy='omit')
        out.append(f'Spearman vegetation vs {v}: rho={r1.statistic:.3f} p={r1.pvalue:.4f} | contamination vs {v}: rho={r2.statistic:.3f} p={r2.pvalue:.4f}')
    for lag in (1, 2, 3):
        rl = c.set_index('ym').rain_mm.reindex(k.ym - lag).values; r1 = stats.spearmanr(k.veg_km2, rl, nan_policy='omit'); out.append(f'vegetation vs rainfall lag {lag}: rho={r1.statistic:.3f} p={r1.pvalue:.4f}')
    ss = k.dropna(subset=['rain_3mo', 'TAVG'])
    def resid(y, X):
        X = np.c_[np.ones(len(X)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
    Xr = np.c_[stats.rankdata(ss.rain_3mo), stats.rankdata(ss.TAVG)]; pr = stats.pearsonr(resid(stats.rankdata(ss.veg_km2), Xr), resid(stats.rankdata(ss.con_km2), Xr))
    out.append(f'partial correlation vegetation-contamination | rain_3mo, TAVG: r={pr.statistic:.3f} p={pr.pvalue:.4f}')
    # monthly climatology (Table 6)
    k['mon'] = k.date.dt.month; clim = k.groupby('mon').agg(n=('veg_km2', 'size'), veg_mean=('veg_km2', 'mean'), veg_med=('veg_km2', 'median'), con_mean=('con_km2', 'mean'), con_med=('con_km2', 'median'))
    cc = c[(c.index >= f'{config.FIRST_MONTH}-01') & (c.index <= f'{config.LAST_MONTH}-01')].copy(); cc['mon'] = cc.index.month
    clim = clim.join(cc.groupby('mon').agg(rain=('rain_mm', 'mean'), tavg=('TAVG', 'mean'), tmax=('TMAX', 'mean'))); clim.to_csv(os.path.join(a.out, 'table6_monthly_climatology.csv'))
    # periodogram (Figure S5)
    periods = np.linspace(2.0, 63.0, 400); fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, v, col, title in ((axes[0], 'veg_km2', G, 'a) Vegetation'), (axes[1], 'con_km2', O, 'b) Contamination')):
        tt = (k.date - m.date.min()).dt.days.values / 30.4375; x = k[v].values.astype(float); p = ls_power(tt, x, periods); r1, (p95, p99) = ar1_levels(tt, x, periods)
        i = np.argmax(p); out.append(f'periodogram {v}: peak {periods[i]:.1f} months (power {p[i]:.3f}; above 95%: {p[i] > p95[i]}; above 99%: {p[i] > p99[i]}); lag-1 r={r1:.2f}')
        ax.plot(periods, p, color=col, lw=1.4, label='Lomb-Scargle power'); ax.plot(periods, p95, color='#777777', lw=0.9, ls='--', label='AR(1) 95% level'); ax.plot(periods, p99, color='k', lw=0.9, ls=':', label='AR(1) 99% level')
        ax.axvline(62, color='k', lw=0.6, alpha=0.4); ax.text(59, 0.004, 'record length', ha='right', va='bottom', fontsize=7, color='#777777', rotation=90)
        ax.set_xscale('log'); ax.xaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 12, 24, 48])); ax.xaxis.set_major_formatter(FixedFormatter(['2', '3', '4', '6', '12', '24', '48'])); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Period (months)'); ax.set_title(f'{title} (lag-1 r = {r1:.2f})', fontsize=9, loc='left'); ax.set_ylim(0, max(p.max(), p99.max()) * 1.18)
        done = []
        for j in np.argsort(p)[::-1]:
            if p[j] > p95[j] and all(abs(periods[j] - periods[d]) > 1.5 for d in done):
                done.append(j); ax.annotate(f'{periods[j]:.1f} mo', (periods[j], p[j]), xytext=(8, 6), textcoords='offset points', fontsize=7.5, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
            if len(done) >= 2: break
    axes[0].set_ylabel('Normalized power'); axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S5_periodogram.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S5_periodogram.eps'), tight=True); plt.close(fig)
    # site-state sensitivity (Table S3)
    labels = ['Recovered', 'Active recovery', 'Transitional', 'Bare/degraded', 'Contaminated']; srows = []
    for name, qq in (('terciles', (1 / 3, 2 / 3)), ('quartiles', (0.25, 0.75)), ('40/60', (0.4, 0.6)), ('median split', (0.5, 0.5))):
        vlo, vhi = k.veg_km2.quantile(qq[0]), k.veg_km2.quantile(qq[1]); clo, chi = k.con_km2.quantile(qq[0]), k.con_km2.quantile(qq[1])
        def cl(v, cc_):
            if v > vhi and cc_ < clo: return 'Recovered'
            if v > vhi and cc_ > chi: return 'Active recovery'
            if v < vlo and cc_ > chi: return 'Contaminated'
            if v < vlo and cc_ < clo: return 'Bare/degraded'
            return 'Transitional'
        s = [cl(v, cc_) for v, cc_ in zip(k.veg_km2, k.con_km2)]; T = np.zeros((5, 5))
        for x, y in zip(s[:-1], s[1:]): T[labels.index(x), labels.index(y)] += 1
        srows.append(dict(scheme=name, **{l: s.count(l) for l in labels}, persistence=np.trace(T) / T.sum()))
    pd.DataFrame(srows).to_csv(os.path.join(a.out, 'tableS3_state_sensitivity.csv'), index=False)
    # Figure 4
    tt_all = (m.date - m.date.min()).dt.days.values / 365.25
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True, gridspec_kw={'height_ratios': [1.0, 1.2, 1.2]})
    ax = axes[0]; ax.bar(m.date, m.rain_mm, width=25, color=B, alpha=0.6, label='CHIRPS rainfall'); ax.set_ylabel('Rainfall (mm/month)'); ax.set_ylim(0, m.rain_mm.max() * 1.6); ax.set_title('a) Monthly rainfall and mean air temperature', fontsize=9, loc='left')
    ax2 = ax.twinx(); ax2.plot(m.date, m.TAVG, color=K, lw=0.9, label='Air temperature'); ax2.set_ylabel('Temperature (°C)'); ax2.spines['top'].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left')
    ax = axes[1]; r = res['veg_km2']; ax.plot(k.date, k.veg_km2, 'o-', color=G, ms=3, lw=1, label='Vegetation class (screened scenes)'); ax.plot(m[m.flag].date, m[m.flag].veg_km2, 'x', color='k', ms=6, label='Excluded scenes')
    ax.plot(m.date, r['intercept'] + r['slope'] * tt_all, '--', color=G, lw=1, label=f"Theil-Sen {r['slope']:+.3f} km²/yr (Mann-Kendall p = {r['p']:.2f})")
    ax.set_ylabel('Vegetation area (km²)'); ax.set_ylim(0, max(m.veg_km2.max(), k.veg_km2.max()) * 1.45); ax.legend(fontsize=7, frameon=False, loc='upper left'); ax.set_title('b) Vegetation class area', fontsize=9, loc='left')
    ax = axes[2]; r = res['con_km2']; ax.plot(k.date, k.con_km2, 'o-', color=O, ms=3, lw=1, label='Contamination class (screened scenes)'); ax.plot(m.date, r['intercept'] + r['slope'] * tt_all, '--', color=O, lw=1, label=f"Theil-Sen {r['slope']:+.3f} km²/yr (Mann-Kendall p = {r['p']:.2f})")
    fl = m[m.flag]; ax.plot(fl.date, np.minimum(fl.con_km2, 15.5), 'x', color='k', ms=6, label='Excluded scenes (clipped at 15.5)')
    ax.set_ylabel('Contamination area (km²)'); ax.set_ylim(0, 23); ax.legend(fontsize=7, frameon=False, loc='upper left'); ax.set_title('c) Contamination class area', fontsize=9, loc='left')
    for ax in axes: ax.axvline(pd.Timestamp(config.REVEGETATION_START), color='k', lw=0.6, ls=':')
    axes[2].set_xlabel('Date'); fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_4_timeseries.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_4_timeseries.eps'), tight=True); plt.close(fig)
    # Figure 5
    y0 = pd.Timestamp(config.REVEGETATION_START).year; phases = [f'2019–{y0 - 1}', f'{y0}–{config.LAST_MONTH[:4]}']
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.2), gridspec_kw={'width_ratios': [1.0, 1.4, 1.1], 'wspace': 0.5})
    ax = axes[0]; bp = ax.boxplot([k[k.season == s].veg_km2.values for s in SEASONS], patch_artist=True, medianprops={'color': 'k'}); [b.set(facecolor=G, alpha=0.4) for b in bp['boxes']]
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(SEASONS, fontsize=8, rotation=45, ha='right', rotation_mode='anchor'); ax.set_ylabel('Vegetation area (km²)'); ax.set_ylim(0, 2.2); ax.set_title('a) Season', fontsize=9, loc='left')
    ax.text(0.97, 0.97, 'Kruskal-Wallis\nH = %.1f, p = %.3f' % (kw_v.statistic, kw_v.pvalue), transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax = axes[1]; sc = ax.scatter(k.rain_3mo, k.veg_km2, c=k.TAVG, cmap='coolwarm', s=18, edgecolor='k', lw=0.3); rs = stats.spearmanr(k.veg_km2, k.rain_3mo, nan_policy='omit')
    ax.set_xlabel('Antecedent 3-month rainfall (mm)'); ax.set_ylabel('Vegetation area (km²)'); ax.set_title('b) Rainfall response', fontsize=9, loc='left'); ax.set_ylim(0, 2.2); cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04); cb.set_label('Air temperature (°C)', fontsize=7, labelpad=2); cb.ax.tick_params(labelsize=7)
    ax.text(0.97, 0.97, 'Spearman ρ = %.2f\n%s' % (rs.statistic, 'p < 0.001' if rs.pvalue < 0.001 else 'p = %.3f' % rs.pvalue), transform=ax.transAxes, ha='right', va='top', fontsize=7)
    ax = axes[2]; bp = ax.boxplot([pre.veg_km2.values, post.veg_km2.values], patch_artist=True, medianprops={'color': 'k'}); [b.set(facecolor=cc_, alpha=0.4) for b, cc_ in zip(bp['boxes'], [K, G])]
    u = stats.mannwhitneyu(pre.veg_km2, post.veg_km2); ax.set_xticks([1, 2]); ax.set_xticklabels(phases, fontsize=7.5); ax.set_xlim(0.35, 2.65); ax.set_ylim(0, 2.2); ax.set_title(f'c) Pre vs post {y0}', fontsize=9, loc='left')
    ax.text(0.97, 0.97, 'Mann-Whitney\nU = %.0f, p = %.2f' % (u.statistic, u.pvalue), transform=ax.transAxes, ha='right', va='top', fontsize=7)
    fig.savefig(os.path.join(a.out, 'Figure_5_seasonal_climate.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_5_seasonal_climate.eps'), tight=True); plt.close(fig)
    # Figure S3: pre/post boxplots for both classes
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, v, col, lab in ((axes[0], 'veg_km2', G, 'Vegetation'), (axes[1], 'con_km2', O, 'Contamination')):
        bp = ax.boxplot([pre[v].values, post[v].values], patch_artist=True, medianprops={'color': 'k'}); [b.set(facecolor=col, alpha=0.4) for b in bp['boxes']]
        u = stats.mannwhitneyu(pre[v], post[v]); rr = 1 - 2 * u.statistic / (len(pre) * len(post)); ax.set_xticks([1, 2]); ax.set_xticklabels([f'{ph}\n(n = {len(d)})' for ph, d in zip(phases, (pre, post))], fontsize=8)
        ax.set_ylabel(f'{lab} area (km²)'); ax.set_title(lab, fontsize=9, loc='left'); ax.set_ylim(top=ax.get_ylim()[1] * 1.15); ax.text(0.97, 0.97, f'U = {u.statistic:.0f}, p = {u.pvalue:.2f}, r = {rr:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S3_prepost.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S3_prepost.eps'), tight=True); plt.close(fig)
    # Figure S4: seasonal boxplots for both classes
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, v, col, lab, kw in ((axes[0], 'veg_km2', G, 'Vegetation', kw_v), (axes[1], 'con_km2', O, 'Contamination', kw_c)):
        bp = ax.boxplot([k[k.season == s][v].values for s in SEASONS], patch_artist=True, medianprops={'color': 'k'}); [b.set(facecolor=col, alpha=0.4) for b in bp['boxes']]
        ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(SEASONS, fontsize=8, rotation=45, ha='right', rotation_mode='anchor'); ax.set_ylabel(f'{lab} area (km²)'); ax.set_title(lab, fontsize=9, loc='left'); ax.set_ylim(top=ax.get_ylim()[1] * 1.15); ax.text(0.97, 0.97, f'H = {kw.statistic:.1f}, p = {kw.pvalue:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S4_seasonal.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S4_seasonal.eps'), tight=True); plt.close(fig)
    # Figure S1: annual mean and SD for the full years
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8)); yy = k[k.year.between(2019, 2023)]
    for ax, v, col, lab in ((axes[0], 'veg_km2', G, 'Vegetation'), (axes[1], 'con_km2', O, 'Contamination')):
        g = yy.groupby('year')[v]; ax.bar(g.mean().index, g.mean().values, yerr=g.std().values, color=col, alpha=0.6, capsize=3); ax.set_ylabel(f'{lab} area (km²)'); ax.set_xlabel('Year'); kwy = stats.kruskal(*[gg.values for _, gg in g]); ax.set_title(f'{lab}, annual mean ± SD', fontsize=9, loc='left'); ax.text(0.97, 0.97, f'H = {kwy.statistic:.1f}, p = {kwy.pvalue:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S1_yearly.png'), dpi=300, bbox_inches='tight'); epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S1_yearly.eps'), tight=True); plt.close(fig)
    open(os.path.join(a.out, 'monthly_series_stats.txt'), 'w', encoding='utf-8').write('\n'.join(out)); print('\n'.join(out)); print('saved to', a.out)

if __name__ == '__main__':
    main()
