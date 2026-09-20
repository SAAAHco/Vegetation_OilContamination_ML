# -*- coding: utf-8 -*-
"""Study constants shared by the analysis scripts (values as stated in the revised manuscript)."""

# Analysis window used for the Landsat HCI record and the HCI validation: KOC map frame, UTM zone 38N (EPSG:32638)
WINDOW_UTM38 = (779000, 3194700, 795000, 3204000)          # xmin, ymin, xmax, ymax (m)
WINDOW_LONLAT = (47.86, 28.846, 48.026, 28.934)              # approximate lon/lat bounds of the same window

# Sentinel-2 composite exports (EOSDA LandViewer): ground pixel size established by SIFT matching (Section 2.2)
COMPOSITE_PX_EW_M = 5.00
COMPOSITE_PX_NS_M = 5.04
COMPOSITE_SHAPE = (2141, 3647)        # rows, cols
VEGMASK_SHAPE = (818, 1395)           # rows, cols of the vegetation product (13.1 m x 13.2 m)
KM2_PER_CONTAMINATION_PIXEL = COMPOSITE_PX_EW_M * COMPOSITE_PX_NS_M / 1e6
KM2_PER_VEGETATION_PIXEL = (COMPOSITE_PX_EW_M * COMPOSITE_SHAPE[1] / VEGMASK_SHAPE[1]) * (COMPOSITE_PX_NS_M * COMPOSITE_SHAPE[0] / VEGMASK_SHAPE[0]) / 1e6
POLYGON_AREA_KM2 = 93.6

# Monthly series
N_SCENES = 63
FIRST_MONTH = '2018-12'
LAST_MONTH = '2024-02'
RADIOMETRIC_Z_LIMIT = 2.0             # scenes with |z| > 2 on the red-minus-blue colour balance are excluded
REVEGETATION_START = '2022-01-01'     # pre/post split

# Landsat HCI record
LANDSAT_CLOUD_MAX = 5.0               # percent
LANDSAT_MIN_VALID_FRACTION = 0.5      # of the window after quality masking
HCI_ENVELOPE_YEARS = (1989, 1990)     # pre-war envelope (5th to 95th percentile of scene medians)
HCI_ANOMALY_LOW = 0.05
HCI_ANOMALY_HIGH = 0.20

# Change classes (Section 2.3)
PERSISTENT_MIN_FREQ = 0.5
REMOVED_MAX_FREQ = 0.2
POLYGON_EDGE_ERODE_PX = 5             # composite edge is classed as contamination in most scenes

# Vegetation index parameters
SAVI_L = 0.5
ARVI_GAMMA = 1.0
VEGETATION_SAVI_THRESHOLD = 0.15
