# Vegetation and contamination monitoring of the southern Greater Burgan Oil Field, Kuwait

Code and derived data for:

> Ashkanani, Z., Mohtar, R., Al-Momin, M., Hetrick, S., Al-Enezi, S., Abdulrahman, R., Albatayneh, R.
> *Deep Learning and Remote Sensing Framework for Assessing Vegetation Recovery in Petroleum-Contaminated Arid Soils Following Large-Scale Remediation.* Journal of Hazardous Materials Advances (revised manuscript HAZADV-D-26-00576).

The repository reproduces every quantitative result of the revised manuscript from public archives and from the
monthly classification masks produced by the CNN. It has three parts:

| Folder | Content |
|---|---|
| `cnn/` | Encoder-decoder CNN, vegetation indices (SAVI, ARVI, HCI), benchmark models, training and evaluation scripts (Sections 2.3, S1.3, S1.4, S1.8). `cnn/config.py` holds the study values. |
| `scripts/` | Numbered analysis scripts that reproduce Figures 3 to 7 and S1 to S8 and Tables 3 to 6, S1 and S3 (Sections 2.2 to 3.4). |
| `data/` | Derived data tables used in the paper (`data/derived/`) and the scene date list. Raw inputs are described below. |

## Data sources

| Input | Source | Access |
|---|---|---|
| Landsat 4, 5, 7, 8, 9 Collection 2 Level-2 surface reflectance, 1989 to 2024 | USGS, via Microsoft Planetary Computer STAC (`landsat-c2-l2`) | public, no account needed for search; signed asset URLs are obtained by `planetary-computer` |
| Sentinel-2 MSI Level-2A, monthly scenes December 2018 to February 2024 | ESA Copernicus, exported through EOS Data Analytics LandViewer as B11/B12/B8A composites; same-day scenes verified in the Planetary Computer `sentinel-2-l2a` collection | public |
| CHIRPS v2.0 rainfall (0.05 degree) | Climate Hazards Center, via the ClimateSERV API | public |
| GHCN-Daily air temperature, Kuwait International Airport (KU000405820) | NOAA NCEI | public |
| Monthly vegetation and contamination masks (63 scenes) | CNN output (this study) | available from the corresponding author on reasonable request, subject to Kuwait Oil Company approval |
| KOC scope-of-work contamination polygons | Kuwait Oil Company Soil Remediation Group map (Figure 1 of the paper) | digitized from the map by `scripts/06_hci_validation_koc.py` |
| Field plots (75), TPH by GC-FID, ASD FieldSpec 4 spectra | Kuwait Oil Company and the authors | on reasonable request, subject to KOC approval |

## Pipeline

All scripts take command-line arguments with defaults that match the folder layout used by the authors; run `python scripts/<name>.py --help`.

| Step | Script | Paper items | Inputs | Outputs |
|---|---|---|---|---|
| 1 | `01_landsat_hci_extract.py` | Section 2.2, 2.4 | Planetary Computer (network) | `hci_landsat_allscenes.csv` (one row per scene: window median, percentiles, anomalous fractions) |
| 2 | `02_hci_summary_figure.py` | Figure 3, Section 3.1 | step 1 output, CHIRPS monthly | `hci_annual_summary.csv`, `Figure_3_HCI_longterm.png/tiff/eps`, cross-sensor offsets |
| 3 | `03_radiometric_screen.py` | Section 2.2, S1.6, Table S1, Figure S7 | monthly composites, scene dates | `composite_radiometry.csv`, `table_S1_acquisitions.csv` (with same-day archive check), `Figure_S7` |
| 4 | `04_monthly_series.py` | Sections 3.2, 3.3; Figures 4, 5, S1, S3, S4, S5; Tables 5, 6, S3 | masks, step 3 output, climate | `monthly_final_km2.csv`, statistics text file, figures |
| 5 | `05_spatial_products.py` | Section 3.4; Figures 6, 7, S8 | masks, step 3 output | `spatial_change_stats.csv`, `fd_multiscale.csv`, `lacunarity_multiscale.csv`, `gradient_profile.csv`, figures |
| 6 | `06_hci_validation_koc.py` | Section S1.3, Figure S6 | Figure 1 map image, Planetary Computer | `hci_roc_vs_koc.csv`, `Figure_S6` |
| 7 | `07_georeference_sift.py` | Section 2.2, S1.2 | one composite + same-day Sentinel-2 scene | pixel size and affine transform (`georef_affine.npy`) |
| 8 | `08_index_sensitivity.py` | Section S1.3 | Sentinel-2 scenes (network) | SAVI L-sensitivity, NDVI and EVI statistics |
| 9 | `09_climate_fetch.py` | Section 2.2, S1.2 | ClimateSERV, NOAA (network) | `chirps_monthly_site.csv`, `airport_monthly_temp.csv`, `climate_monthly_final.csv` |
| 10 | `10_annual_maps.py` | Figure S2, Section S2.2 | masks, step 3 output | `Figure_S2_annual_maps.png/tiff/eps`, `fig_s2_annual_stats.csv` (annual areas at the 25% level, largest patches) |

Every figure script writes PNG (300 dpi) and EPS versions of its figures (TIFF as well where the paper uses it); `scripts/epsexport.py` flattens semi-transparent colours before the EPS is written because PostScript has no transparency.

Constants shared by the scripts (analysis window, pixel sizes, screening threshold, thresholds for change classes) are in `config.py`.

### CNN training (`cnn/`)

```bash
cd cnn
python scripts/train_cnn.py --data_dir <patch folder> --output_dir ./outputs --n_folds 5 --augment --seed 42
python scripts/run_benchmarks.py --data_dir <patch folder>
```

The configuration (`cnn/config.py`) reproduces Table S2 of the paper: 2,500 patches of 256 x 256 pixels split 60/20/20 by
spatial block before augmentation, augmentation of the training set only (factor 8), five-fold spatially blocked
cross-validation, He-normal initialization, Adam (0.001, cosine annealing), batch 32, 100 epochs with early stopping
(patience 15), L2 1e-4, dropout 0.3, gradient clipping 1.0.

## Reproducing the paper from the derived data

If the raw masks and composites are not available, steps 2 and 4 to 5 can be re-run from the tables in `data/derived/`
(`monthly_final_km2.csv` contains the screened monthly class areas; `hci_landsat_allscenes.csv` contains the full Landsat record).

## Environment

Python 3.10 or later. Install with `pip install -r requirements.txt`. Steps 1, 2, 6, 7, 8 and 9 need internet access.
The Landsat extraction (step 1) reads about 1,300 scenes and takes 40 to 60 minutes; signed URLs expire after about an hour, so the script re-signs items in batches.

## Citation

If you use this code or the derived tables, please cite the paper:

Ashkanani, Z., Mohtar, R., Al-Momin, M., Hetrick, S., Al-Enezi, S., Abdulrahman, R., Albatayneh, R. Deep Learning and Remote Sensing Framework for Assessing Vegetation Recovery in Petroleum-Contaminated Arid Soils Following Large-Scale Remediation. Journal of Hazardous Materials Advances (in revision). Repository: https://github.com/SAAAHco/Vegetation_OilContamination_ML (tag `v1.1-revision`).

## Contact

For questions, open an issue in this repository or contact Ashkanani@tamu.edu.

## License

MIT (see `LICENSE`). Landsat data courtesy of the U.S. Geological Survey; Sentinel-2 data contain modified Copernicus Sentinel data; CHIRPS by the Climate Hazards Center, University of California, Santa Barbara; GHCN-Daily by NOAA NCEI.
