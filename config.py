"""
Configuration Template for Vegetation-Contamination Analysis Framework.

This configuration file contains ALL parameters referenced in the methodology.
Users should customize the values marked with [CUSTOMIZE] for their specific study.

Reference:
    Manuscript Section 2: Methodology
    Supplementary Material S1: Detailed Methods
    
IMPORTANT: Replace all [CUSTOMIZE] values with your study-specific parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


# =============================================================================
# STUDY AREA CONFIGURATION [CUSTOMIZE ALL VALUES]
# =============================================================================
@dataclass
class StudyAreaConfig:
    """
    Study area parameters - customize for your site.
    
    Reference: Manuscript Section 2.1, Supplementary Material S1.1
    """
    # Geographic bounds [CUSTOMIZE]
    name: str = "Your_Study_Site"  # e.g., "Greater_Burgan_Oil_Field"
    latitude: float = 0.0  # Center latitude (decimal degrees)
    longitude: float = 0.0  # Center longitude (decimal degrees)
    area_km2: float = 0.0  # Total study area in km²
    
    # Coordinate system [CUSTOMIZE]
    crs_epsg: int = 32639  # UTM Zone - select appropriate for your region
    # Common options: 32632 (UTM 32N Europe), 32610 (UTM 10N Western US), etc.
    
    # Temporal range [CUSTOMIZE]
    start_year: int = 2019
    end_year: int = 2024
    
    # Contamination history [CUSTOMIZE]
    contamination_event_date: str = "1991-01-01"  # Date of contamination event
    remediation_start_date: str = "2019-01-01"  # When remediation began


# =============================================================================
# SATELLITE DATA CONFIGURATION [CUSTOMIZE]
# =============================================================================
@dataclass
class SatelliteConfig:
    """
    Satellite sensor and band configuration.
    
    Reference: Supplementary Material S1.2
    
    Supported sensors (select one or customize for your sensor):
    - Landsat-8 OLI (30m resolution)
    - Landsat-9 OLI-2 (30m resolution)
    - Sentinel-2 MSI (10-20m resolution)
    - Custom hyperspectral sensor
    """
    # Sensor selection [CUSTOMIZE]
    sensor_name: str = "Landsat-8"  # Options: "Landsat-8", "Landsat-9", "Sentinel-2", "Custom"
    spatial_resolution_m: float = 30.0  # Native pixel size in meters
    
    # Band mapping [CUSTOMIZE for your sensor]
    # Maps band names to band numbers or wavelengths
    band_mapping: Dict[str, str] = field(default_factory=lambda: {
        'coastal_aerosol': 'B1',  # ~443nm
        'blue': 'B2',              # ~482nm
        'green': 'B3',             # ~562nm
        'red': 'B4',               # ~655nm
        'nir': 'B5',               # ~865nm (Near-Infrared)
        'swir1': 'B6',             # ~1609nm (Shortwave Infrared 1)
        'swir2': 'B7',             # ~2201nm (Shortwave Infrared 2)
    })
    
    # Bands to use for analysis [CUSTOMIZE]
    analysis_bands: List[str] = field(default_factory=lambda: [
        'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
    ])
    
    # Quality thresholds [CUSTOMIZE based on your data quality]
    max_cloud_cover_percent: float = 10.0  # Maximum acceptable cloud cover
    min_snr: float = 50.0  # Minimum signal-to-noise ratio (Eq. 14)


# =============================================================================
# VEGETATION INDEX CONFIGURATION
# =============================================================================
@dataclass
class VegetationIndexConfig:
    """
    Vegetation index parameters.
    
    Reference: 
        Manuscript Section 2.3
        Supplementary Material S1.3, Equations 1-3
    """
    # SAVI parameters (Equation 2) [CUSTOMIZE]
    savi_L: float = 0.5  # Soil brightness correction factor
    # L = 0: high vegetation density (equivalent to NDVI)
    # L = 0.5: intermediate vegetation density (recommended for most cases)
    # L = 1.0: low vegetation density
    
    # ARVI parameters (Equation 1) [CUSTOMIZE]
    arvi_gamma: float = 1.0  # Atmospheric correction coefficient
    # gamma = 1.0: standard aerosol conditions
    # gamma = 0.5: heavy aerosol/dust conditions
    # Adjust based on atmospheric conditions in your study area
    
    # HCI parameters (Equation 3) [CUSTOMIZE]
    # HCI = (ρ_2100 - ρ_660) / (ρ_2100 + ρ_660)
    hci_band_2100nm: str = 'swir2'  # Band approximating 2100nm
    hci_band_660nm: str = 'red'     # Band approximating 660nm
    
    # HCI-TPH calibration [CUSTOMIZE with your ground truth data]
    # TPH (mg/kg) = hci_slope * HCI + hci_intercept
    hci_tph_slope: float = 0.0  # [CUSTOMIZE] Calibration slope from your data
    hci_tph_intercept: float = 0.0  # [CUSTOMIZE] Calibration intercept
    hci_calibration_r2: float = 0.0  # [CUSTOMIZE] R² of your calibration
    
    # Detection thresholds [CUSTOMIZE based on your calibration]
    hci_detection_limit: float = 0.10  # Minimum HCI for contamination detection
    hci_detection_tph_mgkg: float = 2500  # Corresponding TPH concentration


# =============================================================================
# CNN MODEL CONFIGURATION
# =============================================================================
@dataclass  
class CNNConfig:
    """
    CNN encoder-decoder architecture parameters.
    
    Reference:
        Manuscript Section 2.3
        Supplementary Material S1.4, Equations 6-10
    """
    # Input configuration [CUSTOMIZE]
    input_height: int = 256  # Patch height in pixels
    input_width: int = 256   # Patch width in pixels
    input_channels: int = 6  # Number of spectral bands
    
    # Encoder architecture [CUSTOMIZE]
    # Number of filters at each encoder level
    encoder_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    kernel_size: Tuple[int, int] = (3, 3)  # Convolution kernel size
    pool_size: Tuple[int, int] = (2, 2)    # Max pooling size
    
    # Decoder architecture (mirrors encoder with skip connections)
    use_skip_connections: bool = True
    
    # Output configuration [CUSTOMIZE]
    n_classes: int = 4  # Number of output classes
    # Example classes: [Background, Contaminated, Recovering, Recovered]
    class_names: List[str] = field(default_factory=lambda: [
        'Background', 'Contaminated', 'Recovering', 'Recovered'
    ])
    output_activation: str = 'softmax'  # 'softmax' for classification, 'linear' for regression
    
    # Regularization [CUSTOMIZE based on your dataset size]
    dropout_rate: float = 0.3  # Dropout rate for fully connected layers
    l2_weight_decay: float = 1e-4  # L2 regularization strength
    gradient_clip_norm: float = 1.0  # Gradient clipping threshold


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
@dataclass
class TrainingConfig:
    """
    Model training parameters.
    
    Reference: Supplementary Material S1.4
    """
    # Dataset configuration [CUSTOMIZE]
    n_training_patches: int = 2500  # Number of original training patches
    validation_split: float = 0.2   # Fraction for validation
    test_split: float = 0.2         # Fraction for final testing
    
    # Data augmentation [CUSTOMIZE]
    use_augmentation: bool = True
    augmentation_factor: int = 8  # Multiplier for effective dataset size
    rotation_angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_range: float = 0.15  # ±15% brightness adjustment
    noise_std: float = 0.01  # Gaussian noise standard deviation
    
    # Optimizer configuration [CUSTOMIZE]
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Learning rate schedule [CUSTOMIZE]
    use_lr_schedule: bool = True
    lr_schedule_type: str = 'cosine'  # Options: 'cosine', 'step', 'exponential'
    
    # Training loop [CUSTOMIZE]
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Cross-validation [CUSTOMIZE]
    n_folds: int = 5  # Number of cross-validation folds
    use_spatial_blocking: bool = True  # Spatially-blocked CV to prevent data leakage


# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================
@dataclass
class TemporalConfig:
    """
    Temporal and spectral analysis parameters.
    
    Reference:
        Supplementary Material S1.5, Equations 11-13
    """
    # FFT Analysis (Equation 12) [CUSTOMIZE]
    fft_sampling_rate: float = 12.0  # Samples per year (monthly = 12)
    fft_freq_min: float = 0.1  # Minimum frequency (cycles/year)
    fft_freq_max: float = 12.0  # Maximum frequency (cycles/year)
    
    # Monte Carlo significance testing [CUSTOMIZE]
    fft_n_permutations: int = 1000  # Number of permutations for significance
    fft_significance_level: float = 0.95  # Confidence threshold
    
    # CUSUM Analysis (Equation 13) [CUSTOMIZE]
    cusum_k_factor: float = 0.5  # Reference value as multiple of σ
    cusum_h_factor: float = 4.0  # Detection threshold as multiple of σ
    cusum_bootstrap_iterations: int = 1000
    
    # Binary mask thresholding (Equation 11) [CUSTOMIZE]
    # These should be optimized for your specific study area
    vegetation_threshold: float = 0.15  # SAVI threshold for vegetation
    contamination_threshold: float = 0.10  # HCI threshold for contamination


# =============================================================================
# SPATIAL ANALYSIS CONFIGURATION
# =============================================================================
@dataclass
class SpatialConfig:
    """
    Spatial pattern analysis parameters.
    
    Reference:
        Supplementary Material S1.7
    """
    # Fractal Analysis (Equation 15) [CUSTOMIZE]
    # Box sizes for box-counting method (in pixels)
    fractal_box_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])
    # Corresponding sizes in meters depend on your pixel resolution
    
    # Lacunarity Analysis [CUSTOMIZE]
    lacunarity_box_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 11, 21, 31])
    
    # Connected component labeling [CUSTOMIZE]
    connectivity: int = 8  # 4 or 8 connectivity
    min_patch_pixels: int = 2  # Minimum patch size to include
    
    # Gradient Analysis [CUSTOMIZE]
    gradient_max_distance_m: float = 10000.0  # Maximum distance from contamination
    gradient_interval_m: float = 500.0  # Sampling interval


# =============================================================================
# RECOVERY STATE CLASSIFICATION [CUSTOMIZE]
# =============================================================================
@dataclass
class RecoveryStateConfig:
    """
    Ecosystem recovery state definitions.
    
    Reference: Supplementary Material S1.7.5
    
    IMPORTANT: These thresholds should be calibrated for your specific
    vegetation communities and contamination types.
    """
    # State definitions using SAVI and HCI thresholds [CUSTOMIZE]
    states: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'recovered': {
            'savi_min': 0.30,
            'hci_max': 0.10,
            'min_duration_months': 6
        },
        'active_recovery': {
            'savi_min': 0.15,
            'savi_max': 0.30,
            'hci_max': 0.25,
            'trend': 'positive'
        },
        'transitional': {
            'savi_min': 0.10,
            'savi_max': 0.20,
            'hci_min': 0.15,
            'hci_max': 0.35
        },
        'bare_degraded': {
            'savi_max': 0.10,
            'hci_max': 0.15
        },
        'contaminated': {
            'hci_min': 0.35
        }
    })


# =============================================================================
# BENCHMARK MODEL CONFIGURATION
# =============================================================================
@dataclass
class BenchmarkConfig:
    """
    Benchmark model configurations for comparison.
    
    Reference: Supplementary Material S1.8
    """
    # Random Forest [CUSTOMIZE]
    rf_n_estimators: int = 500
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    
    # Support Vector Machine [CUSTOMIZE]
    svm_kernel: str = 'rbf'
    svm_C: float = 10.0
    svm_gamma: float = 0.01
    
    # XGBoost [CUSTOMIZE]
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    
    # Deep Learning benchmarks [CUSTOMIZE]
    pretrained_lr: float = 0.0001  # Learning rate for fine-tuning
    scratch_lr: float = 0.001  # Learning rate for training from scratch


# =============================================================================
# QUALITY ASSESSMENT CONFIGURATION
# =============================================================================
@dataclass
class QualityConfig:
    """
    Quality assessment and uncertainty parameters.
    
    Reference: Supplementary Material S1.6, Equation 14
    """
    # Radiometric quality [CUSTOMIZE]
    min_snr: float = 50.0  # Minimum signal-to-noise ratio
    
    # Geometric accuracy [CUSTOMIZE]
    max_rmse_pixels: float = 0.3  # Maximum geometric RMSE
    min_gcps: int = 50  # Minimum ground control points
    
    # Field data quality ratings [CUSTOMIZE]
    q1_max_days: int = 0  # Same day
    q2_max_days: int = 3  # ±3 days
    q3_max_days: int = 7  # ±7 days
    min_homogeneity_q2: float = 0.80  # 80% homogeneous
    min_homogeneity_q3: float = 0.60  # 60% homogeneous
    
    # Uncertainty estimates [CUSTOMIZE based on your conditions]
    uncertainty_optimal_percent: float = 7.2  # Under clear conditions
    uncertainty_dust_storm_percent: float = 15.3  # During dust events
    dust_storm_aod_threshold: float = 1.0  # AOD threshold for dust storms


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
@dataclass
class OutputConfig:
    """
    Output and reporting configuration.
    """
    # File formats
    raster_format: str = 'GeoTIFF'
    vector_format: str = 'GeoPackage'
    report_format: str = 'markdown'
    
    # Output directories (relative to project root)
    output_dir: str = './outputs'
    model_dir: str = './outputs/models'
    figure_dir: str = './outputs/figures'
    report_dir: str = './outputs/reports'


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================
@dataclass
class Config:
    """
    Master configuration combining all components.
    
    Usage:
        config = Config()
        # Customize for your study:
        config.study_area.name = "My_Study_Site"
        config.study_area.latitude = 35.0
        # etc.
    """
    study_area: StudyAreaConfig = field(default_factory=StudyAreaConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    vegetation_index: VegetationIndexConfig = field(default_factory=VegetationIndexConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    recovery_states: RecoveryStateConfig = field(default_factory=RecoveryStateConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []
        
        # Check for placeholder values
        if self.study_area.latitude == 0.0:
            issues.append("WARNING: Study area latitude not set (still 0.0)")
        if self.study_area.longitude == 0.0:
            issues.append("WARNING: Study area longitude not set (still 0.0)")
        if self.study_area.area_km2 == 0.0:
            issues.append("WARNING: Study area size not set (still 0.0)")
        if self.study_area.name == "Your_Study_Site":
            issues.append("WARNING: Study area name not customized")
        if self.vegetation_index.hci_tph_slope == 0.0:
            issues.append("WARNING: HCI-TPH calibration slope not set")
            
        # Validate ranges
        if not 0 <= self.vegetation_index.savi_L <= 1:
            issues.append("ERROR: SAVI L factor must be between 0 and 1")
        if self.training.validation_split + self.training.test_split >= 1.0:
            issues.append("ERROR: validation_split + test_split must be < 1.0")
            
        return issues
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"\nStudy Area: {self.study_area.name}")
        print(f"  Location: {self.study_area.latitude}°N, {self.study_area.longitude}°E")
        print(f"  Area: {self.study_area.area_km2} km²")
        print(f"  Period: {self.study_area.start_year}-{self.study_area.end_year}")
        print(f"\nSatellite: {self.satellite.sensor_name}")
        print(f"  Resolution: {self.satellite.spatial_resolution_m}m")
        print(f"  Bands: {len(self.satellite.analysis_bands)}")
        print(f"\nCNN Architecture:")
        print(f"  Input: {self.cnn.input_height}x{self.cnn.input_width}x{self.cnn.input_channels}")
        print(f"  Filters: {self.cnn.encoder_filters}")
        print(f"  Classes: {self.cnn.n_classes}")
        print(f"\nTraining:")
        print(f"  Patches: {self.training.n_training_patches}")
        print(f"  Augmentation: {self.training.augmentation_factor}x")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  CV Folds: {self.training.n_folds}")
        print("=" * 60)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':
    # Create default configuration
    config = Config()
    
    # Validate and show warnings
    issues = config.validate()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Print summary
    config.print_summary()
    
    print("\n[!] Remember to customize all [CUSTOMIZE] parameters for your study!")
