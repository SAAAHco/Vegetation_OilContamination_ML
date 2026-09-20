#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 3: scene-level radiometric consistency screen for the monthly composites (Section 2.2, S1.6, Table S1, Figure S7).
Inputs: a folder with composites named <n>.tiff (n = 1..N, newest first as exported) and a dates file with lines
'DDMMYY n' (the authors' Dates.txt) or a CSV with columns image,date. Optional --archive-check confirms same-day
Sentinel-2 / Landsat scenes in the Planetary Computer archive."""
import argparse, os, re, sys, datetime as dt
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
Image.MAX_IMAGE_PIXELS = None

def read_dates(path):
    if path.lower().endswith('.csv'):
        d = pd.read_csv(path); return {int(r.image): pd.to_datetime(r.date).date() for _, r in d.iterrows()}
    txt = open(path, encoding='utf-8', errors='ignore').read()
    pairs = re.findall(r'^(\d{6})\s+(\d{1,2})\s*$', txt, flags=re.M)
    return {int(n): dt.date(2000 + int(s[4:6]), int(s[2:4]), int(s[0:2])) for s, n in pairs}

def archive_check(dates):
    import planetary_computer as pc, pystac_client
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    bbox = list(config.WINDOW_LONLAT); out = {}
    for n, d in dates.items():
        d0 = (d - dt.timedelta(days=1)).isoformat(); d1 = (d + dt.timedelta(days=1)).isoformat()
        ls = [i for i in cat.search(collections=['landsat-c2-l2'], bbox=bbox, datetime=f'{d0}/{d1}').items() if i.properties['datetime'][:10] == d.isoformat()]
        s2 = [i for i in cat.search(collections=['sentinel-2-l2a'], bbox=bbox, datetime=f'{d0}/{d1}').items() if i.properties['datetime'][:10] == d.isoformat()]
        if s2: sensor = 'Sentinel-2 MSI (' + '/'.join(sorted(set(i.properties.get('platform', '') for i in s2))) + ')'; cc = min(i.properties.get('eo:cloud_cover', np.nan) for i in s2)
        elif ls: sensor = 'Landsat ' + '/'.join(sorted(set(i.properties.get('platform', '').replace('landsat-', '') for i in ls))); cc = min(i.properties.get('eo:cloud_cover', np.nan) for i in ls)
        else: sensor, cc = 'no same-day archive scene', np.nan
        out[n] = (sensor, cc); print(n, d, sensor, flush=True)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--composites', required=True); ap.add_argument('--dates', required=True)
    ap.add_argument('--n', type=int, default=config.N_SCENES); ap.add_argument('--archive-check', action='store_true')
    ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dates = read_dates(a.dates); rows = []
    for n in range(1, a.n + 1):
        arr = np.array(Image.open(os.path.join(a.composites, f'{n}.tiff')).convert('RGB')).astype(float); m = arr.sum(axis=2) > 30; px = arr[m]
        rows.append(dict(image=n, date=dates[n], R=px[:, 0].mean(), G=px[:, 1].mean(), B=px[:, 2].mean()))
    d = pd.DataFrame(rows).sort_values('date'); d['RminusB'] = d.R - d.B; d['z_RminusB'] = (d.RminusB - d.RminusB.mean()) / d.RminusB.std()
    d['flag'] = d.z_RminusB.abs() > config.RADIOMETRIC_Z_LIMIT
    if a.archive_check:
        chk = archive_check(dates); d['sensor'] = d.image.map(lambda n: chk[n][0]); d['scene_cloud'] = d.image.map(lambda n: chk[n][1])
    d.to_csv(os.path.join(a.out, 'composite_radiometry.csv'), index=False)
    t = d[['image', 'date'] + (['sensor', 'scene_cloud'] if a.archive_check else []) + ['z_RminusB', 'flag']].copy(); t['used'] = np.where(t.flag, 'no', 'yes')
    t.to_csv(os.path.join(a.out, 'table_S1_acquisitions.csv'), index=False)
    print('excluded scenes:', d.loc[d.flag, ['image', 'date', 'z_RminusB']].to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import epsexport
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); ax.plot(pd.to_datetime(d.date), d.z_RminusB, 'o-', color='#444444', ms=3, lw=0.8)
    for y in (config.RADIOMETRIC_Z_LIMIT, -config.RADIOMETRIC_Z_LIMIT): ax.axhline(y, color='r', lw=0.8, ls='--')
    for j, (_, r) in enumerate(d[d.flag].iterrows()):
        dy = 8 if r.z_RminusB > 0 else -12; dx = -34 if j % 2 == 0 else 6
        ax.annotate(pd.Timestamp(r.date).strftime('%b %Y'), (pd.Timestamp(r.date), r.z_RminusB), xytext=(dx, dy), textcoords='offset points', fontsize=7, color='r', ha='left')
    ax.set_ylim(-3.2, 3.9); ax.set_ylabel('Colour-balance z-score (R − B)'); ax.set_xlabel('Date'); ax.set_title('Scene-level radiometric consistency screen (|z| > %g excluded)' % config.RADIOMETRIC_Z_LIMIT, fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'Figure_S7_radiometric_screen.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(a.out, 'Figure_S7_radiometric_screen.tiff'), dpi=1000, pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    epsexport.save_eps(fig, os.path.join(a.out, 'Figure_S7_radiometric_screen.eps'), tight=True); plt.close(fig); print('saved to', a.out)

if __name__ == '__main__':
    main()
