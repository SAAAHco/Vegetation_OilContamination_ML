"""
Preprocessing module for satellite image data.

This module provides functions for loading, preprocessing, and augmenting
Landsat-8 multispectral imagery for oil contamination detection.
"""

from .data_loader import (
    load_satellite_bands,
    load_training_patches,
    create_dataset,
    normalize_bands,
    stack_bands
)

from .data_augmentation import (
    create_augmentation_pipeline,
    apply_augmentation,
    geometric_augmentation,
    photometric_augmentation
)

from .patch_extraction import (
    extract_patches,
    create_patch_dataset,
    stratified_patch_sampling
)

__all__ = [
    'load_satellite_bands',
    'load_training_patches',
    'create_dataset',
    'normalize_bands',
    'stack_bands',
    'create_augmentation_pipeline',
    'apply_augmentation',
    'geometric_augmentation',
    'photometric_augmentation',
    'extract_patches',
    'create_patch_dataset',
    'stratified_patch_sampling'
]
