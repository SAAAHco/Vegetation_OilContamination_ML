#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 9: monthly CHIRPS rainfall over the analysis polygon (ClimateSERV API) and monthly mean and maximum air
temperature from the Kuwait International Airport GHCN-Daily record (Section 2.2, S1.2)."""
import argparse, io, json, os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', default='12/01/2018'); ap.add_argument('--end', default='02/29/2024'); ap.add_argument('--station', default='KU000405820'); ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    lon0, lat0, lon1, lat1 = config.WINDOW_LONLAT
    poly = {'type': 'Polygon', 'coordinates': [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}
    base = 'https://climateserv.servirglobal.net/api/'
    r = requests.get(base + 'submitDataRequest/', params={'datatype': 0, 'begintime': a.start, 'endtime': a.end, 'intervaltype': 0, 'operationtype': 5, 'geometry': json.dumps(poly)}, timeout=120); rid = json.loads(r.text)[0]
    for _ in range(120):
        p = requests.get(base + 'getDataRequestProgress/', params={'id': rid}, timeout=60).json()
        if float(p[0]) >= 100: break
        time.sleep(5)
    d = requests.get(base + 'getDataFromRequest/', params={'id': rid}, timeout=120).json()
    daily = pd.DataFrame([(x['date'], x['value']['avg']) for x in d['data']], columns=['date', 'mm']); daily['date'] = pd.to_datetime(daily.date); daily.to_csv(os.path.join(a.out, 'chirps_daily_site.csv'), index=False)
    monthly = daily.set_index('date').mm.resample('MS').sum().rename('rain_mm'); monthly.to_csv(os.path.join(a.out, 'chirps_monthly_site.csv'))
    print('CHIRPS annual totals (mm):', monthly.groupby(monthly.index.year).sum().round(0).to_dict())
    g = pd.read_csv(f'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{a.station}.csv', low_memory=False); g['DATE'] = pd.to_datetime(g['DATE']); g = g[g.DATE >= '2018-01-01']
    mt = g.set_index('DATE')[['TAVG', 'TMAX']].resample('MS').mean() / 10.0; mt.to_csv(os.path.join(a.out, 'airport_monthly_temp.csv'))
    clim = pd.concat([monthly, mt], axis=1).sort_index(); clim.to_csv(os.path.join(a.out, 'climate_monthly_final.csv')); print('saved climate tables to', a.out)

if __name__ == '__main__':
    main()
