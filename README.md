# Vegetation-Contamination Analysis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A reusable template for AI-driven vegetation monitoring in contaminated environments using deep learning and satellite imagery.**

This repository provides a complete, customizable framework implementing the methodology described in:

> **"Enhancing Vegetation Coverage Monitoring through Artificial Intelligence-Integrated Remote Sensing"**
> 
> *Remote Sensing of Environment* (submitted)

---

## 🎯 Overview

This framework integrates:
- **CNN encoder-decoder architecture** for vegetation/contamination classification
- **Vegetation indices** (ARVI, SAVI, HCI) optimized for arid environments
- **Temporal analysis** (FFT, CUSUM) for detecting recovery patterns
- **Spatial analysis** (fractal dimension, lacunarity, Markov chains) for landscape characterization
- **Comprehensive benchmarking** against traditional ML and deep learning models

---

## 📋 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SAAAHco/vegetation-contamination-framework.git
cd vegetation-contamination-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Customize Configuration

**IMPORTANT**: Before using this framework, customize `config.py` for your study:

```python
from config import Config

# Load default configuration
config = Config()

# ============================================
# CUSTOMIZE THESE FOR YOUR STUDY AREA
# ============================================

# Study area (Section 2.1)
config.study_area.name = "Your_Study_Site"
config.study_area.latitude = 35.0  # Your latitude
config.study_area.longitude = -120.0  # Your longitude
config.study_area.area_km2 = 100.0  # Your study area size
config.study_area.crs_epsg = 32610  # Your UTM zone

# Satellite configuration (Section S1.2)
config.satellite.sensor_name = "Sentinel-2"  # Your sensor
config.satellite.spatial_resolution_m = 10.0

# Vegetation indices calibration (Section S1.3)
# *** CALIBRATE WITH YOUR GROUND TRUTH DATA ***
config.vegetation_index.hci_tph_slope = 12847.0  # From YOUR regression
config.vegetation_index.hci_tph_intercept = 1243.0

# Recovery state thresholds (Section S1.7.5)
# *** ADJUST FOR YOUR VEGETATION COMMUNITIES ***
config.recovery_states.states['recovered']['savi_min'] = 0.30
config.recovery_states.states['contaminated']['hci_min'] = 0.35

# Validate configuration
issues = config.validate()
for issue in issues:
    print(issue)
```

### 3. Run Analysis

```bash
# Train CNN model
python scripts/train_cnn.py --data_dir /path/to/your/data --output_dir ./outputs

# Run benchmark comparisons
python scripts/run_benchmarks.py --data_dir /path/to/your/data
```

---

## 📁 Repository Structure

```
vegetation-contamination-framework/
├── config.py                    # 🔧 CUSTOMIZE THIS FIRST
├── requirements.txt
├── LICENSE
├── README.md
│
├── models/                      # Core ML models
│   ├── cnn_encoder_decoder.py   # CNN architecture (Eq. 6-10)
│   └── vegetation_indices.py    # ARVI, SAVI, HCI (Eq. 1-3)
│
├── benchmarks/                  # Comparison models (S1.8)
│   ├── traditional_ml.py        # RF, SVM, XGBoost
│   └── deep_learning.py         # VGG-16, ResNet-50, U-Net, DeepLabV3+
│
├── analysis/                    # Analysis modules
│   ├── temporal_analysis.py     # FFT, CUSUM (Eq. 11-13)
│   └── spatial_analysis.py      # Fractal, lacunarity (Eq. 15)
│
├── preprocessing/               # Data preparation
│   ├── data_loader.py           # Satellite image I/O
│   ├── data_augmentation.py     # Training augmentation
│   └── patch_extraction.py      # Patch extraction
│
├── utils/                       # Utilities
│   ├── metrics.py               # Evaluation metrics (Eq. 4-5, 14)
│   └── visualization.py         # Plotting functions
│
├── scripts/                     # Executable scripts
│   ├── train_cnn.py             # Training pipeline
│   └── run_benchmarks.py        # Benchmark comparisons
│
└── notebooks/                   # Jupyter notebooks
    └── 01_demonstration.ipynb   # Interactive demo
```

---

## 🔬 Methodology Implementation

This framework implements all methods described in the manuscript and supplementary material:

### Vegetation Indices (Section 2.3, S1.3)

| Index | Equation | Reference | Parameters to Customize |
|-------|----------|-----------|------------------------|
| **ARVI** | `(NIR - RB) / (NIR + RB)` where `RB = Red - γ(Blue - Red)` | Eq. 1 | `γ` (default: 1.0) |
| **SAVI** | `((NIR - Red) / (NIR + Red + L)) × (1 + L)` | Eq. 2 | `L` (default: 0.5) |
| **HCI** | `(ρ₂₁₀₀ - ρ₆₆₀) / (ρ₂₁₀₀ + ρ₆₆₀)` | Eq. 3 | Band mapping, TPH calibration |

### CNN Architecture (Section 2.3, S1.4)

| Component | Specification | Customizable |
|-----------|--------------|--------------|
| Encoder | 5 blocks (32→64→128→256→512 filters) | Filter counts |
| Decoder | Transposed conv with skip connections | Yes |
| Input | 256×256 pixels, 6 bands | Patch size, bands |
| Output | Pixel-wise classification | Number of classes |
| Training | Adam, lr=0.001, cosine annealing | All hyperparameters |

### Temporal Analysis (S1.5)

| Method | Equation | Key Parameters |
|--------|----------|----------------|
| **FFT** | `X(k) = Σₙ x(n)·e^(-j2πkn/N)` | Freq range: 0.1-12 cycles/yr |
| **CUSUM** | `CUSUMₙ = max(0, CUSUMₙ₋₁ + xₙ - μ - k)` | k=0.5σ, h=4σ |

### Spatial Analysis (S1.7)

| Method | Equation | Application |
|--------|----------|-------------|
| **Fractal Dimension** | `FD = lim(log N(ε) / log(1/ε))` | Pattern complexity |
| **Lacunarity** | `Λ(r) = (σ²/μ²) + 1` | Spatial heterogeneity |
| **Markov Chain** | Transition probability matrix | Recovery state dynamics |

---

## 🔧 Customization Guide

### Step 1: Configure Study Area

Edit `config.py` to set your geographic bounds, coordinate system, and temporal range.

### Step 2: Calibrate Vegetation Indices

**Critical**: The HCI-TPH relationship must be calibrated with YOUR ground truth data:

```python
# Collect field samples with:
# - GPS coordinates
# - TPH concentration (lab analysis)
# - Coincident satellite imagery

# Perform linear regression: TPH = slope × HCI + intercept
# Update in config.py:
config.vegetation_index.hci_tph_slope = YOUR_SLOPE
config.vegetation_index.hci_tph_intercept = YOUR_INTERCEPT
config.vegetation_index.hci_calibration_r2 = YOUR_R2
```

### Step 3: Define Recovery States

Adjust SAVI/HCI thresholds for your vegetation communities:

```python
# Example for grassland ecosystem
config.recovery_states.states = {
    'recovered': {'savi_min': 0.40, 'hci_max': 0.05},
    'active_recovery': {'savi_min': 0.20, 'savi_max': 0.40, 'hci_max': 0.15},
    'transitional': {'savi_min': 0.10, 'savi_max': 0.25, 'hci_min': 0.10, 'hci_max': 0.30},
    'bare_degraded': {'savi_max': 0.10, 'hci_max': 0.10},
    'contaminated': {'hci_min': 0.30}
}
```

### Step 4: Prepare Training Data

```python
from preprocessing.data_loader import SatelliteDataLoader
from preprocessing.patch_extraction import PatchExtractor

# Load your satellite imagery
loader = SatelliteDataLoader(
    data_dir='/path/to/your/imagery',
    bands=config.satellite.analysis_bands
)
images, labels = loader.load_dataset()

# Extract patches
extractor = PatchExtractor(
    patch_size=config.cnn.input_height,
    overlap=0.25,
    min_valid_ratio=0.8
)
X, y = extractor.extract_patches_from_dataset(images, labels)
```

---

## 📊 Expected Performance

When properly calibrated, this framework achieves results comparable to the methodology benchmarks:

| Model | Accuracy | F1 Score | Cohen's κ |
|-------|----------|----------|-----------|
| SVM (RBF) | ~79% | ~0.76 | ~0.58 |
| Random Forest | ~82% | ~0.79 | ~0.62 |
| XGBoost | ~84% | ~0.81 | ~0.65 |
| VGG-16 | ~82% | ~0.79 | ~0.61 |
| ResNet-50 | ~84% | ~0.82 | ~0.67 |
| U-Net | ~87% | ~0.85 | ~0.73 |
| DeepLabV3+ | ~86% | ~0.84 | ~0.71 |
| **CNN (This framework)** | **~89%** | **~0.88** | **~0.76** |

*Note: Actual results depend on your specific dataset, calibration quality, and study area characteristics.*

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{author2024vegetation,
  title={Enhancing Vegetation Coverage Monitoring through Artificial 
         Intelligence-Integrated Remote Sensing},
  author={[Zainab Ashkanani]},
  journal={Remote Sensing of Environment},
  year={2026},
  note={Submitted}
}
```

---

## 📚 References

Key methodological references:

1. **ARVI**: Kaufman, Y.J., & Tanre, D. (1992). *IEEE Trans. Geosci. Remote Sens.*, 30, 261-270.
2. **SAVI**: Huete, A.R. (1988). *Remote Sensing of Environment*, 25, 295-309.
3. **HCI**: Kühn, F., et al. (2004). *Int. J. Remote Sens.*, 25, 2467-2473.
4. **CUSUM**: Ygorra, B., et al. (2021). *Int. J. Appl. Earth Obs. Geoinf.*, 103, 102532.
5. **Fractal Analysis**: Li, H., et al. (2009). *Landscape Ecology*, 24, 291-302.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Commit changes (`git commit -am 'Add new analysis method'`)
4. Push to branch (`git push origin feature/new-method`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ❓ FAQ

**Q: Can I use this for non-oil contamination studies?**
A: Yes! The framework is applicable to any vegetation stress monitoring. Adjust the HCI or create custom indices for your contaminant type.

**Q: What satellite sensors are supported?**
A: The framework works with any multispectral sensor. Configure band mappings in `config.py` for your sensor (Landsat-8/9, Sentinel-2, MODIS, etc.).

**Q: How much ground truth data do I need?**
A: Minimum ~50 field samples for HCI-TPH calibration, ~75+ plots for robust validation. More data improves calibration accuracy.

**Q: Can I use different vegetation indices?**
A: Yes! Add custom indices in `models/vegetation_indices.py` following the existing patterns.

---

## 📧 Contact

For questions or support, please open an issue or contact [Ashkanani@tamu.edu].
