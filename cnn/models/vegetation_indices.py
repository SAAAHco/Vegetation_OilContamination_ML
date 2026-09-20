"""
Vegetation Indices Calculation Module

This module implements the vegetation and contamination indices used in the study:
- ARVI (Atmospherically Resistant Vegetation Index)
- SAVI (Soil-Adjusted Vegetation Index)
- HCI (Hyperspectral Contamination Index)
- NDVI (Normalized Difference Vegetation Index) - for comparison

References:
    - Manuscript Section 2.2 (Methods - Vegetation Index Computation)
    - Supplementary Material Section S1.3 (Spectral Analysis and Index Computation)
    - Kaufman & Tanre (1992) - ARVI
    - Huete (1988) - SAVI
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


def calculate_arvi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Calculate Atmospherically Resistant Vegetation Index (ARVI).
    
    ARVI provides atmospheric resistance by using the blue band to correct 
    for atmospheric scattering effects on the red band.
    
    Formula from manuscript (Equation 1):
        ARVI = (NIR - RB) / (NIR + RB)
        where RB = Red - γ(Blue - Red)
    
    Reference: Kaufman, Y.J., & Tanre, D. (1992). IEEE TGRS, 30, 261-270.
    
    Args:
        nir: Near-infrared band (Landsat-8 Band 5 / Sentinel-2 B08)
        red: Red band (Landsat-8 Band 4 / Sentinel-2 B04)
        blue: Blue band (Landsat-8 Band 2 / Sentinel-2 B02)
        gamma: Atmospheric correction coefficient (default=1.0)
              "γ = 1.0 provides optimal atmospheric resistance for most 
              environmental conditions"
              
    Returns:
        ARVI values ranging from -1 to 1
    """
    # Convert to float to prevent integer overflow
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    blue = blue.astype(np.float64)
    
    # Calculate atmospherically corrected red band (RB)
    # RB = Red - γ(Blue - Red)
    rb = red - gamma * (blue - red)
    
    # Calculate ARVI
    # ARVI = (NIR - RB) / (NIR + RB)
    denominator = nir + rb
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        arvi = np.where(
            denominator != 0,
            (nir - rb) / denominator,
            0
        )
    
    # Clip to valid range [-1, 1]
    arvi = np.clip(arvi, -1, 1)
    
    return arvi


def calculate_savi(
    nir: np.ndarray,
    red: np.ndarray,
    L: float = 0.5
) -> np.ndarray:
    """
    Calculate Soil-Adjusted Vegetation Index (SAVI).
    
    SAVI minimizes soil brightness influences from spectral vegetation indices 
    using a soil brightness correction factor (L).
    
    Formula from manuscript (Equation 2):
        SAVI = ((NIR - Red) / (NIR + Red + L)) × (1 + L)
    
    Reference: Huete, A.R. (1988). Remote Sensing of Environment, 25, 295-309.
    
    Args:
        nir: Near-infrared band (Landsat-8 Band 5 / Sentinel-2 B08)
        red: Red band (Landsat-8 Band 4 / Sentinel-2 B04)
        L: Soil brightness correction factor
           "L = 0.5 is appropriate for intermediate vegetation cover conditions 
           typical of the semi-arid Burgan oil field region"
           - L = 0 for very high vegetation cover
           - L = 1 for very low vegetation cover
           - L = 0.5 for intermediate conditions (default)
           
    Returns:
        SAVI values ranging from -1 to 1
    """
    # Convert to float
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    
    # Calculate SAVI
    # SAVI = ((NIR - Red) / (NIR + Red + L)) × (1 + L)
    denominator = nir + red + L
    
    with np.errstate(divide='ignore', invalid='ignore'):
        savi = np.where(
            denominator != 0,
            ((nir - red) / denominator) * (1 + L),
            0
        )
    
    # Clip to valid range [-1, 1]
    savi = np.clip(savi, -1, 1)
    
    return savi


def calculate_hci(
    swir1: np.ndarray,
    swir2: np.ndarray,
    red: np.ndarray
) -> np.ndarray:
    """
    Calculate Hyperspectral Contamination Index (HCI).
    
    HCI amplifies contaminant spectral signatures by leveraging the spectral 
    response of hydrocarbons in SWIR wavelengths.
    
    Formula from Supplementary Material (Equation 3):
        HCI = (ρ²¹⁰⁰ - ρ⁶⁶⁰) / (ρ²¹⁰⁰ + ρ⁶⁶⁰)
    
    Simplified implementation using available bands:
        HCI = (SWIR2 - Red) / (SWIR2 + Red)
    
    Higher HCI values indicate greater hydrocarbon presence.
    Calibration: TPH (mg/kg) = 12,847 × HCI + 1,243 (R² = 0.78)
    
    Args:
        swir1: Short-wave infrared band 1 (Landsat-8 Band 6 / Sentinel-2 B11)
        swir2: Short-wave infrared band 2 (Landsat-8 Band 7 / Sentinel-2 B12)
        red: Red/visible band (Landsat-8 Band 4 / Sentinel-2 B04)
        
    Returns:
        HCI values (theoretical range -1 to 1)
    """
    # Convert to float
    swir2 = swir2.astype(np.float64)
    red = red.astype(np.float64)
    
    # Calculate HCI
    denominator = swir2 + red
    
    with np.errstate(divide='ignore', invalid='ignore'):
        hci = np.where(
            denominator != 0,
            (swir2 - red) / denominator,
            0
        )
    
    return hci


def calculate_hci_alternative(
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray
) -> np.ndarray:
    """
    Alternative HCI calculation used in the original code.
    
    Formula: HCI = (NIR - (SWIR1 + SWIR2)) / 2
    
    This formula captures the spectral contrast between vegetation-sensitive 
    NIR and hydrocarbon-sensitive SWIR bands.
    
    Args:
        nir: Near-infrared band (B08 or B08A)
        swir1: Short-wave infrared band 1 (B11)
        swir2: Short-wave infrared band 2 (B12)
        
    Returns:
        HCI values
    """
    nir = nir.astype(np.float64)
    swir1 = swir1.astype(np.float64)
    swir2 = swir2.astype(np.float64)
    
    hci = (nir - (swir1 + swir2)) / 2
    
    return hci


def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray
) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    Standard vegetation index for comparison purposes.
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir: Near-infrared band
        red: Red band
        
    Returns:
        NDVI values ranging from -1 to 1
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    
    denominator = nir + red
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0
        )
    
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi


def calculate_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0
) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index (EVI).
    
    EVI is optimized for high biomass regions and reduces atmospheric influences.
    
    Formula: EVI = G × (NIR - Red) / (NIR + C1×Red - C2×Blue + L)
    
    Args:
        nir: Near-infrared band
        red: Red band
        blue: Blue band
        G: Gain factor (default 2.5)
        C1: Coefficient for aerosol resistance (default 6.0)
        C2: Coefficient for aerosol resistance (default 7.5)
        L: Canopy background adjustment (default 1.0)
        
    Returns:
        EVI values (typically -1 to 1)
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    blue = blue.astype(np.float64)
    
    denominator = nir + C1 * red - C2 * blue + L
    
    with np.errstate(divide='ignore', invalid='ignore'):
        evi = np.where(
            denominator != 0,
            G * (nir - red) / denominator,
            0
        )
    
    return evi


def calculate_enhanced_vegetation_index(
    arvi: np.ndarray,
    savi: np.ndarray,
    weights: Optional[Tuple[float, float]] = None,
    ground_truth: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate enhanced vegetation index through weighted fusion of ARVI and SAVI.
    
    From Supplementary Material Section S1.3:
    "To enhance detection accuracy in contaminated areas, we developed a novel 
    index fusion approach. This method combines SAVI and ARVI values through a 
    weighted averaging procedure, with weights determined through an iterative 
    optimization process using ground-truth data."
    
    Args:
        arvi: ARVI values
        savi: SAVI values
        weights: Tuple of (arvi_weight, savi_weight). If None, uses default (0.4, 0.6)
                 or optimizes using ground_truth if provided.
        ground_truth: Optional ground truth data for weight optimization
        
    Returns:
        Enhanced vegetation index values
    """
    if weights is not None:
        w_arvi, w_savi = weights
    elif ground_truth is not None:
        # Optimize weights using ground truth
        w_arvi, w_savi = _optimize_fusion_weights(arvi, savi, ground_truth)
    else:
        # Default weights based on empirical testing
        # "The resulting enhanced vegetation index demonstrated improved sensitivity 
        # to vegetation stress caused by hydrocarbon contamination"
        w_arvi = 0.4
        w_savi = 0.6
    
    # Normalize weights
    total = w_arvi + w_savi
    w_arvi /= total
    w_savi /= total
    
    # Weighted fusion
    enhanced_vi = w_arvi * arvi + w_savi * savi
    
    return enhanced_vi


def _optimize_fusion_weights(
    arvi: np.ndarray,
    savi: np.ndarray,
    ground_truth: np.ndarray,
    n_iterations: int = 100
) -> Tuple[float, float]:
    """
    Optimize fusion weights using iterative search.
    
    Args:
        arvi: ARVI values
        savi: SAVI values
        ground_truth: Ground truth vegetation health data
        n_iterations: Number of iterations for optimization
        
    Returns:
        Tuple of optimized (arvi_weight, savi_weight)
    """
    best_corr = -1
    best_weights = (0.5, 0.5)
    
    for i in range(n_iterations):
        w_arvi = i / n_iterations
        w_savi = 1 - w_arvi
        
        fused = w_arvi * arvi + w_savi * savi
        
        # Calculate correlation with ground truth
        # Flatten arrays for correlation
        fused_flat = fused.flatten()
        gt_flat = ground_truth.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(fused_flat) | np.isnan(gt_flat))
        if np.sum(mask) > 10:
            corr = np.corrcoef(fused_flat[mask], gt_flat[mask])[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_weights = (w_arvi, w_savi)
    
    return best_weights


def classify_contamination(
    hci: np.ndarray,
    threshold: float = 0.10
) -> np.ndarray:
    """
    Classify pixels as contaminated based on HCI threshold.
    
    From manuscript: "Detection limits were established at HCI > 0.10, 
    corresponding to approximately 2,500 mg/kg TPH"
    
    Args:
        hci: HCI values
        threshold: Classification threshold (default 0.10)
        
    Returns:
        Binary mask (1 = contaminated, 0 = non-contaminated)
    """
    return (hci > threshold).astype(np.int32)


def classify_recovery_state(
    savi: np.ndarray,
    hci: np.ndarray,
    months_stable: int = 6
) -> np.ndarray:
    """
    Classify ecosystem recovery states based on SAVI and HCI values.
    
    From Supplementary Material Section S1.7.5:
    "Five ecosystem states were defined:
    - Recovered: SAVI>0.30, HCI<0.10, >6 months
    - Active Recovery: 0.15<SAVI<0.30, HCI<0.25, positive trend
    - Transitional: 0.10<SAVI<0.20, 0.15<HCI<0.35
    - Bare/Degraded: SAVI<0.10, HCI<0.15
    - Contaminated: HCI>0.35"
    
    Args:
        savi: SAVI values
        hci: HCI values
        months_stable: Number of months stable (for Recovered classification)
        
    Returns:
        Classification array:
            0 = Contaminated
            1 = Bare/Degraded
            2 = Transitional
            3 = Active Recovery
            4 = Recovered
    """
    states = np.zeros_like(savi, dtype=np.int32)
    
    # Contaminated: HCI > 0.35
    contaminated = hci > 0.35
    states[contaminated] = 0
    
    # Bare/Degraded: SAVI < 0.10, HCI < 0.15
    bare = (savi < 0.10) & (hci < 0.15) & ~contaminated
    states[bare] = 1
    
    # Transitional: 0.10 < SAVI < 0.20, 0.15 < HCI < 0.35
    transitional = (savi > 0.10) & (savi < 0.20) & (hci > 0.15) & (hci < 0.35)
    states[transitional] = 2
    
    # Active Recovery: 0.15 < SAVI < 0.30, HCI < 0.25
    active_recovery = (savi > 0.15) & (savi < 0.30) & (hci < 0.25) & ~transitional
    states[active_recovery] = 3
    
    # Recovered: SAVI > 0.30, HCI < 0.10
    recovered = (savi > 0.30) & (hci < 0.10)
    states[recovered] = 4
    
    return states


def calculate_all_indices(
    bands: dict,
    calculate_enhanced: bool = True
) -> dict:
    """
    Calculate all vegetation indices from multispectral bands.
    
    Args:
        bands: Dictionary with band arrays. Expected keys:
               - 'blue' or 'B02': Blue band
               - 'green' or 'B03': Green band
               - 'red' or 'B04': Red band
               - 'nir' or 'B08': Near-infrared band
               - 'swir1' or 'B11': SWIR band 1
               - 'swir2' or 'B12': SWIR band 2
        calculate_enhanced: Whether to calculate enhanced fusion index
        
    Returns:
        Dictionary with calculated indices
    """
    # Map band names
    blue = bands.get('blue', bands.get('B02'))
    green = bands.get('green', bands.get('B03'))
    red = bands.get('red', bands.get('B04'))
    nir = bands.get('nir', bands.get('B08', bands.get('B08A')))
    swir1 = bands.get('swir1', bands.get('B11'))
    swir2 = bands.get('swir2', bands.get('B12'))
    
    results = {}
    
    # Calculate NDVI
    if nir is not None and red is not None:
        results['NDVI'] = calculate_ndvi(nir, red)
    
    # Calculate SAVI
    if nir is not None and red is not None:
        results['SAVI'] = calculate_savi(nir, red, L=0.5)
    
    # Calculate ARVI
    if nir is not None and red is not None and blue is not None:
        results['ARVI'] = calculate_arvi(nir, red, blue, gamma=1.0)
    
    # Calculate EVI
    if nir is not None and red is not None and blue is not None:
        results['EVI'] = calculate_evi(nir, red, blue)
    
    # Calculate HCI
    if swir1 is not None and swir2 is not None and red is not None:
        results['HCI'] = calculate_hci(swir1, swir2, red)
    
    # Calculate alternative HCI
    if nir is not None and swir1 is not None and swir2 is not None:
        results['HCI_alt'] = calculate_hci_alternative(nir, swir1, swir2)
    
    # Calculate enhanced vegetation index
    if calculate_enhanced and 'ARVI' in results and 'SAVI' in results:
        results['Enhanced_VI'] = calculate_enhanced_vegetation_index(
            results['ARVI'], results['SAVI']
        )
    
    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Vegetation Indices Module")
    print("=" * 50)
    
    # Create synthetic band data
    np.random.seed(42)
    shape = (256, 256)
    
    # Simulate reflectance values (0-10000 for Landsat)
    bands = {
        'blue': np.random.uniform(500, 2000, shape).astype(np.float32),
        'green': np.random.uniform(600, 2500, shape).astype(np.float32),
        'red': np.random.uniform(400, 3000, shape).astype(np.float32),
        'nir': np.random.uniform(1500, 5000, shape).astype(np.float32),
        'swir1': np.random.uniform(1000, 4000, shape).astype(np.float32),
        'swir2': np.random.uniform(500, 3000, shape).astype(np.float32),
    }
    
    # Calculate all indices
    indices = calculate_all_indices(bands)
    
    # Print statistics
    for name, values in indices.items():
        print(f"\n{name}:")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
