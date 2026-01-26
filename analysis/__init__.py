"""
Temporal and spatial analysis modules for oil contamination recovery monitoring.

This module provides advanced signal processing (FFT, CUSUM, wavelet) and
landscape pattern analysis (fractal dimension, lacunarity, Markov chain)
as described in the supplementary materials.

Reference:
    Supplementary Material S1.5: Temporal and Spectral Analysis
    Supplementary Material S1.7: Spatial Pattern Analysis
"""

from .temporal_analysis import (
    fft_analysis,
    cusum_analysis,
    wavelet_analysis,
    seasonal_decomposition,
    recovery_phase_analysis,
    vegetation_contamination_coupling,
    monte_carlo_significance_test,
    trend_analysis
)

from .spatial_analysis import (
    fractal_dimension_box_counting,
    lacunarity_analysis,
    markov_chain_analysis,
    landscape_metrics,
    gradient_analysis,
    spatial_autocorrelation_morans_i,
    recovery_state_classification,
    patch_dynamics_analysis
)

__all__ = [
    # Temporal Analysis
    'fft_analysis',
    'cusum_analysis',
    'wavelet_analysis',
    'seasonal_decomposition',
    'recovery_phase_analysis',
    'vegetation_contamination_coupling',
    'monte_carlo_significance_test',
    'trend_analysis',
    # Spatial Analysis
    'fractal_dimension_box_counting',
    'lacunarity_analysis',
    'markov_chain_analysis',
    'landscape_metrics',
    'gradient_analysis',
    'spatial_autocorrelation_morans_i',
    'recovery_state_classification',
    'patch_dynamics_analysis'
]
