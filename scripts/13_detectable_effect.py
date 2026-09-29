#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 13: minimum detectable effect of the pre/post revegetation comparison (Section 3.2). Bootstraps the screened
pre-2022 vegetation class areas (34 scenes) to estimate the power of the two-sided Mann-Whitney test (alpha = 0.05,
24 post-period scenes) for multiplicative and additive sustained increases, and gives the asymptotic minimum detectable
rank-biserial effect at 80 % power. Writes power_numbers.json."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--monthly', required=True, help='step 4 output (monthly_final_km2.csv)')
ap.add_argument('--nsim', type=int, default=6000); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default='.'); args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
m = pd.read_csv(args.monthly, parse_dates=['date']).sort_values('date'); k = m[~m.flag]
pre = k[(k.date >= '2019-01-01') & (k.date < config.REVEGETATION_START)].veg_km2.values; post = k[k.date >= config.REVEGETATION_START].veg_km2.values
rng = np.random.default_rng(args.seed)
def power(transform):
    hits = 0
    for _ in range(args.nsim):
        a = rng.choice(pre, len(pre), replace=True); b = transform(rng.choice(pre, len(post), replace=True))
        hits += stats.mannwhitneyu(a, b, alternative='two-sided').pvalue < 0.05
    return hits / args.nsim
out = dict(n_pre=int(len(pre)), n_post=int(len(post)), pre_median=float(np.median(pre)), post_median=float(np.median(post)), observed_ratio=float(np.median(post) / np.median(pre)),
           observed_p=float(stats.mannwhitneyu(pre, post, alternative='two-sided').pvalue))
for f in (1.9, 2.0, 2.5, 3.0): out[f'power_factor_{f}'] = power(lambda x, f=f: x * f)
for d in (0.15, 0.20, 0.25): out[f'power_add_{d}_km2'] = power(lambda x, d=d: x + d)
n1, n2 = len(pre), len(post); sig = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12); out['mde_rank_biserial_80pct'] = float((1.96 + 0.8416) * sig / (n1 * n2 / 2))
json.dump(out, open(os.path.join(args.out, 'power_numbers.json'), 'w'), indent=1)
for kk, v in out.items(): print(f'{kk}: {v:.3f}' if isinstance(v, float) else f'{kk}: {v}')
