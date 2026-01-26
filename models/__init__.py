"""
Deep learning models for oil contamination detection.

This module provides the CNN encoder-decoder architecture and vegetation index
calculations as described in the manuscript.

Reference:
    Manuscript Section 2.3: Deep Learning Architecture
    Supplementary Material S1.4: Model Implementation Details
"""

from .cnn_encoder_decoder import (
    encoder_block,
    decoder_block,
    build_cnn_encoder_decoder,
    build_classification_model,
    build_regression_model,
    train_model_with_cross_validation
)

from .vegetation_indices import (
    calculate_arvi,
    calculate_savi,
    calculate_hci,
    calculate_ndvi,
    calculate_evi,
    calculate_enhanced_vi,
    calculate_all_indices,
    tph_from_hci,
    classify_recovery_state
)

__all__ = [
    # CNN Encoder-Decoder
    'encoder_block',
    'decoder_block',
    'build_cnn_encoder_decoder',
    'build_classification_model',
    'build_regression_model',
    'train_model_with_cross_validation',
    # Vegetation Indices
    'calculate_arvi',
    'calculate_savi',
    'calculate_hci',
    'calculate_ndvi',
    'calculate_evi',
    'calculate_enhanced_vi',
    'calculate_all_indices',
    'tph_from_hci',
    'classify_recovery_state'
]
