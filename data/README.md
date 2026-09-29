# Data

`derived/` contains the tables produced by the pipeline and used in the paper:

| File | Content | Paper item |
|---|---|---|
| `hci_landsat_allscenes.csv` | HCI window statistics for every Landsat scene, 1989 to 2024 (1,317 scenes) | Figure 3, Section 3.1 |
| `hci_annual_summary.csv` | Annual medians, quartiles, anomalous fractions | Figure 3 |
| `hci_roc_vs_koc.csv` | HCI validation against the KOC polygons | Figure S4, Section S1.3 |
| `composite_radiometry.csv`, `table_S1_acquisitions.csv` | Scene-level colour balance, sensor confirmed from the archive, screening flag | Table S1, Figure S5 |
| `monthly_final_km2.csv` | Monthly vegetation and contamination class areas (km²) with climate covariates and the screening flag | Figures 4, 5, S1, S3; Tables 5, S4 |
| `monthly_climatology.csv`, `seasonal_stats_km2.csv`, `yearly_stats_km2.csv` | Aggregates of the screened series | Tables 5, S4 |
| `recovery_states_sensitivity.csv` | Site-state classification under four threshold schemes | Table S3 |
| `spatial_change_stats.csv` | Persistence and change-class areas | Figure 6, Section 3.4 |
| `fd_multiscale.csv`, `lacunarity_multiscale.csv` | Fractal dimension by scale range and lacunarity by box size, per scene, on the native grids (superseded by the common-grid files) | step 5 |
| `fd_common_grid.csv`, `lacunarity_common_grid.csv`, `fd_common_numbers.json` | The same metrics with both classes on the 13.1 m vegetation grid (59 screened scenes) | Figure 7, Section 3.4, S1.7.4 |
| `l2a_check.csv` | Export-based vegetation class area against SAVI > 0.15 on Level-2A reflectance, 20 screened scenes | Figure S7, Sections 2.4, S2.8 |
| `power_numbers.json` | Bootstrap power of the pre/post comparison and the minimum detectable rank-biserial effect | Section 3.2 |
| `gradient_profile.csv` | Vegetation density against distance from contamination | Figure S6 |
| `chirps_monthly_site.csv`, `airport_monthly_temp.csv`, `climate_monthly_final.csv` | Rainfall and temperature | Figure 4a |

`scene_dates_raw.txt` is the acquisition-date list of the 63 monthly composites (DDMMYY and scene number).

Raw inputs not included here because of size or data-sharing terms: the 63 Sentinel-2 composites and the 126 classification masks
(available from the corresponding author on reasonable request, subject to Kuwait Oil Company approval), the KOC map (Figure 1 of the paper),
and the field plot data.
- `fig_s2_annual_stats.csv`: per-year areas of both classes at the 25% detection-frequency level and at least one detection, and the largest contamination patch (Figure S2, script 10).
