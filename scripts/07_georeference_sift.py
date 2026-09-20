#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 7: ground pixel size and affine transform of an exported composite by SIFT matching against the same-day
Sentinel-2 Level-2A scene (Section 2.2, S1.2). Prints the pixel size in metres and the polygon area in km2."""
import argparse, math, os, sys
import numpy as np, cv2
from PIL import Image
import planetary_computer as pc, pystac_client, rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
Image.MAX_IMAGE_PIXELS = None

def to8(x, lo=2, hi=98):
    a, b = np.percentile(x[x > 0], [lo, hi]); return np.clip((x - a) / (b - a) * 255, 0, 255).astype('uint8')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--composite', required=True); ap.add_argument('--date', required=True, help='YYYY-MM-DD of the composite')
    ap.add_argument('--bbox', default='47.80,28.80,48.08,28.96', help='search/reference bbox lon0,lat0,lon1,lat1')
    ap.add_argument('--band', default='B04'); ap.add_argument('--out', default='.'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    bbox = [float(v) for v in a.bbox.split(',')]; res = 0.0001; W = int(round((bbox[2] - bbox[0]) / res)); H = int(round((bbox[3] - bbox[1]) / res)); tr = from_origin(bbox[0], bbox[3], res, res)
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    items = list(cat.search(collections=['sentinel-2-l2a'], bbox=bbox, datetime=f'{a.date}/{a.date}').items()); items.sort(key=lambda i: i.properties.get('eo:cloud_cover', 99)); it = items[0]
    with rasterio.open(it.assets[a.band].href) as src:
        sb = transform_bounds('EPSG:4326', src.crs, *bbox); win = from_bounds(*sb, transform=src.transform); arr = src.read(1, window=win).astype('float32'); wt = src.window_transform(win)
        ref = np.zeros((H, W), 'float32'); reproject(arr, ref, src_transform=wt, src_crs=src.crs, dst_transform=tr, dst_crs='EPSG:4326', resampling=Resampling.bilinear)
    ref8 = cv2.resize(to8(ref), (W * 2, H * 2), interpolation=cv2.INTER_CUBIC); ref8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(ref8)
    comp = np.array(Image.open(a.composite).convert('RGB')); poly = (comp.sum(axis=2) > 30).astype('uint8'); Hc, Wc = poly.shape
    cg = cv2.resize(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), (Wc // 2, Hc // 2), interpolation=cv2.INTER_AREA); cm = cv2.erode(cv2.resize(poly, (Wc // 2, Hc // 2), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8))
    cg = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cg)
    sift = cv2.SIFT_create(nfeatures=20000); kc, dc = sift.detectAndCompute(cg, cm); kr, dr = sift.detectAndCompute(ref8, None)
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(dc, dr, k=2) if m.distance < 0.75 * n.distance]
    src = np.float32([kc[m.queryIdx].pt for m in good]); dst = np.float32([kr[m.trainIdx].pt for m in good])
    # two-stage fit as used for the paper: similarity (RANSAC) to select inliers, then a full affine on the inliers
    S, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0); keep = inl.ravel() == 1; ninl = int(keep.sum())
    A, inl2 = cv2.estimateAffine2D(src[keep], dst[keep], method=cv2.RANSAC, ransacReprojThreshold=4.0)
    def comp_to_ll(x, y):
        xc, yc = x / 2, y / 2; u = A[0, 0] * xc + A[0, 1] * yc + A[0, 2]; v = A[1, 0] * xc + A[1, 1] * yc + A[1, 2]; return bbox[0] + u * res / 2, bbox[3] - v * res / 2
    lon0, lat1 = comp_to_ll(0, 0); lon1, lat0 = comp_to_ll(Wc, Hc); clat = (lat0 + lat1) / 2; mlon = 111320 * math.cos(math.radians(clat)); mlat = 110950
    pE = (lon1 - lon0) / Wc * mlon; pN = (lat1 - lat0) / Hc * mlat
    print(f'{it.id}: {len(good)} good matches, {ninl} RANSAC inliers; pixel {pE:.2f} m (E-W) x {pN:.2f} m (N-S); polygon {poly.sum() * pE * pN / 1e6:.1f} km2; bbox lon {lon0:.5f}-{lon1:.5f} lat {lat0:.5f}-{lat1:.5f}')
    np.save(os.path.join(a.out, 'georef_affine.npy'), A)

if __name__ == '__main__':
    main()
