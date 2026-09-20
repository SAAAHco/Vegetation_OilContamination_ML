#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 2: annual summary of the Landsat HCI record, cross-sensor offsets, trend test, and Figure 3 (Section 3.1)."""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import epsexport
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def mann_kendall(x):
    n = len(x); s = sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18; z = (s - np.sign(s)) / np.sqrt(var); return z, 2 * (1 - stats.norm.cdf(abs(z)))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenes', default='hci_landsat_allscenes.csv'); ap.add_argument('--chirps', default='chirps_monthly_site.csv')
    ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    O, B, K = '#D55E00', '#0072B2', '#444444'
    d = pd.read_csv(a.scenes, parse_dates=['date']).sort_values('date')
    d = d[d.valid_frac >= config.LANDSAT_MIN_VALID_FRACTION].copy(); d['year'] = d.date.dt.year
    d['anom'] = d.frac_lt_005 + d.frac_gt_020
    pre = d[d.year.between(*config.HCI_ENVELOPE_YEARS)]; env_lo, env_hi, base = pre.hci_p5.median(), pre.hci_p95.median(), pre.hci_med.median()
    g = d.groupby('year').agg(n=('hci_med', 'size'), med=('hci_med', 'median'), q1=('hci_med', lambda s: s.quantile(.25)), q3=('hci_med', lambda s: s.quantile(.75)),
                              anom=('anom', 'median'), anom_q1=('anom', lambda s: s.quantile(.25)), anom_q3=('anom', lambda s: s.quantile(.75)),
                              lt0=('frac_lt_0', 'median'), gt020=('frac_gt_020', 'median'), platforms=('platform', lambda s: '/'.join(sorted(set(x.replace('landsat-', 'L') for x in s)))))
    g.to_csv(os.path.join(a.out, 'hci_annual_summary.csv'))
    print(f'scenes used {len(d)}; pre-war baseline median {base:.3f}, envelope {env_lo:.3f} to {env_hi:.3f}')
    for lo, hi, yrs in (('landsat-5', 'landsat-7', (1999, 2011)), ('landsat-7', 'landsat-8', (2013, 2021)), ('landsat-8', 'landsat-9', (2022, 2024))):
        ov = d[d.year.between(*yrs)].groupby(['year', 'platform']).hci_med.median().unstack()
        if lo in ov and hi in ov:
            diff = (ov[hi] - ov[lo]).dropna(); print(f'{hi} minus {lo}: mean {diff.mean():+.4f} sd {diff.std():.4f} n {len(diff)}')
    post = g[g.index >= 1994]; z, p = mann_kendall(post.med.values); ts = stats.theilslopes(post.med.values, post.index.values)
    print(f'1994 onward annual median: Mann-Kendall z={z:.2f} p={p:.3f}; Theil-Sen {ts[0]:+.5f} per year')
    rec = d[d.date >= f'{config.FIRST_MONTH}-01']
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), gridspec_kw={'height_ratios': [1.2, 1.0, 1.1]})
    ax = axes[0]; ax.fill_between(g.index, g.q1, g.q3, color=O, alpha=0.25, label='Interquartile range of scene medians')
    ax.plot(g.index, g.med, 'o-', color=O, ms=3.5, lw=1.2, label='Annual median HCI (window)')
    ax.axhspan(env_lo, env_hi, color=K, alpha=0.12, zorder=0.6, label='Pre-war envelope (5th to 95th percentile)'); ax.axvspan(1991, 1991.99, color='k', alpha=0.08, zorder=0.5)
    ax.set_ylabel('HCI (dimensionless)'); ax.set_title('a) Annual Landsat HCI over the analysis window', fontsize=9, loc='left'); ax.legend(fontsize=7, frameon=False, loc='upper right')
    for y, n in zip(g.index, g.n): ax.text(y, ax.get_ylim()[0] + 0.005, str(n), ha='center', va='bottom', fontsize=5, color=K)
    ax = axes[1]; ax.fill_between(g.index, g.anom_q1 * 100, g.anom_q3 * 100, color=B, alpha=0.25)
    ax.plot(g.index, g.anom * 100, 'o-', color=B, ms=3.5, lw=1.2, label=f'HCI < {config.HCI_ANOMALY_LOW:.2f} or > {config.HCI_ANOMALY_HIGH:.2f} (anomalous)')
    ax.plot(g.index, g.lt0 * 100, 's--', color=K, ms=2.5, lw=0.9, label='HCI < 0 (liquid-hydrocarbon-like)')
    ax.set_ylabel('Fraction of window (%)'); ax.set_yscale('symlog', linthresh=1); ax.set_title('b) Anomalous-HCI fraction of the window', fontsize=9, loc='left'); ax.legend(fontsize=7, frameon=False, loc='upper right'); ax.set_xlabel('Year')
    ax = axes[2]; ax.plot(rec.date, rec.hci_med, 'o', color=O, ms=3, label='Scene median HCI')
    if os.path.exists(a.chirps):
        ch = pd.read_csv(a.chirps, index_col=0, parse_dates=True).iloc[:, 0]
        ax2 = ax.twinx(); ax2.bar(ch.index, ch.values, width=25, color=B, alpha=0.4, label='CHIRPS rainfall'); ax2.set_ylabel('Rainfall (mm/month)'); ax2.spines['top'].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    else:
        ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_ylabel('HCI (dimensionless)'); ax.set_xlabel('Date'); ax.set_title('c) Scene-level HCI and rainfall over the monthly record', fontsize=9, loc='left')
    ax.set_xlim(pd.Timestamp(f'{config.FIRST_MONTH}-01'), pd.Timestamp(f'{config.LAST_MONTH}-01') + pd.DateOffset(months=1))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_3_HCI_longterm.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(a.out, 'Figure_3_HCI_longterm.tiff'), dpi=1000, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    epsexport.save_eps(fig, os.path.join(a.out, 'Figure_3_HCI_longterm.eps')); plt.close(fig)
    print('saved Figure 3 and hci_annual_summary.csv in', a.out)

if __name__ == '__main__':
    main()
