#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1: Hydrocarbon Contamination Index (HCI = (SWIR2 - red)/(SWIR2 + red)) from every Landsat Collection 2
Level-2 scene over the analysis window (Section 2.2 and 2.4 of the paper). One row per scene.
Scenes are searched year by year so that signed asset URLs never expire during the run."""
import argparse, os, sys, time
import numpy as np, pandas as pd
import planetary_computer as pc, pystac_client, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def read_scene(item, win_ll):
    """Return window statistics of HCI for one STAC item. Quality mask: fill, cloud, cirrus (not shadow)."""
    with rasterio.open(item.assets['qa_pixel'].href) as src:
        qa = src.read(1, window=from_bounds(*transform_bounds('EPSG:4326', src.crs, *win_ll), transform=src.transform))
    with rasterio.open(item.assets['red'].href) as src:
        red = src.read(1, window=from_bounds(*transform_bounds('EPSG:4326', src.crs, *win_ll), transform=src.transform)).astype('float64')
    with rasterio.open(item.assets['swir22'].href) as src:
        sw = src.read(1, window=from_bounds(*transform_bounds('EPSG:4326', src.crs, *win_ll), transform=src.transform)).astype('float64')
    h = min(qa.shape[0], red.shape[0], sw.shape[0]); w = min(qa.shape[1], red.shape[1], sw.shape[1])
    qa, red, sw = qa[:h, :w], red[:h, :w], sw[:h, :w]
    bad = (qa & 1) | (qa & 8) | (qa & 4)
    r = red * 0.0000275 - 0.2; s = sw * 0.0000275 - 0.2
    ok = (bad == 0) & (red > 0) & (sw > 0) & (r > -0.05) & (s > -0.05) & (r < 1) & (s < 1)
    hci = (s - r) / (s + r + 1e-9); v = hci[ok]
    if v.size == 0: return None
    return dict(n_valid=int(ok.sum()), valid_frac=float(ok.mean()), red_med=float(np.median(r[ok])), swir2_med=float(np.median(s[ok])),
                hci_mean=float(v.mean()), hci_med=float(np.median(v)), hci_sd=float(v.std()),
                hci_p5=float(np.percentile(v, 5)), hci_p10=float(np.percentile(v, 10)), hci_p25=float(np.percentile(v, 25)),
                hci_p75=float(np.percentile(v, 75)), hci_p90=float(np.percentile(v, 90)), hci_p95=float(np.percentile(v, 95)),
                frac_lt_0=float((v < 0).mean()), frac_lt_005=float((v < 0.05).mean()), frac_gt_010=float((v > 0.10).mean()),
                frac_gt_015=float((v > 0.15).mean()), frac_gt_020=float((v > 0.20).mean()))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=int, default=1989); ap.add_argument('--end', type=int, default=2024)
    ap.add_argument('--cloud', type=float, default=config.LANDSAT_CLOUD_MAX)
    ap.add_argument('--row', default='040', help='WRS-2 row to keep')
    ap.add_argument('--out', default='hci_landsat_allscenes.csv')
    a = ap.parse_args()
    win_ll = transform_bounds('EPSG:32638', 'EPSG:4326', *config.WINDOW_UTM38)
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    rows = []
    for year in range(a.start, a.end + 1):
        items = list(cat.search(collections=['landsat-c2-l2'], bbox=list(win_ll), datetime=f'{year}-01-01/{year}-12-31',
                                query={'eo:cloud_cover': {'lt': a.cloud}}).items())
        items = [i for i in items if i.properties.get('landsat:wrs_row') == a.row]
        items.sort(key=lambda i: i.properties['datetime'])
        for it in items:
            p = it.properties
            try:
                st = read_scene(it, win_ll)
                if st is None: continue
                rows.append(dict(id=it.id, date=p['datetime'][:10], platform=p.get('platform'), path=p.get('landsat:wrs_path'),
                                 cloud=p.get('eo:cloud_cover'), sun_el=p.get('view:sun_elevation'), **st))
            except Exception as e:
                print('ERR', it.id, e, flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(year, 'scenes so far', len(rows), flush=True)
    print('DONE', len(rows), '->', a.out)

if __name__ == '__main__':
    main()
