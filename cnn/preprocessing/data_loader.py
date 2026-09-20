"""
Data loading utilities for Landsat-8 multispectral satellite imagery.

This module provides functions for loading and preprocessing satellite data
for the oil contamination vegetation analysis model.

References:
    - Manuscript Section 2.1: Study Area and Data Collection
    - Supplementary Material S1.1: Satellite Data Acquisition

Functions:
    load_satellite_bands: Load individual spectral bands
    load_training_patches: Load pre-extracted training patches
    create_dataset: Create TensorFlow dataset for training
    normalize_bands: Apply radiometric normalization
    stack_bands: Stack multiple bands into a single array
"""

import os
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import warnings

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not installed. Some functionality may be limited.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Dataset creation will be limited.")


# Band specifications for Landsat-8 OLI
LANDSAT8_BANDS = {
    'B02': {'name': 'Blue', 'wavelength': 482, 'resolution': 30},
    'B03': {'name': 'Green', 'wavelength': 562, 'resolution': 30},
    'B04': {'name': 'Red', 'wavelength': 655, 'resolution': 30},
    'B08': {'name': 'NIR', 'wavelength': 865, 'resolution': 30},
    'B11': {'name': 'SWIR1', 'wavelength': 1609, 'resolution': 30},
    'B12': {'name': 'SWIR2', 'wavelength': 2201, 'resolution': 30},
}


def load_satellite_bands(
    data_dir: str,
    band_names: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
    file_pattern: str = '{band}.tif'
) -> Dict[str, np.ndarray]:
    """
    Load satellite spectral bands from GeoTIFF files.
    
    Args:
        data_dir: Directory containing the band files
        band_names: List of band names to load (default: 6 bands for model)
        file_pattern: File naming pattern with {band} placeholder
        
    Returns:
        Dictionary mapping band names to numpy arrays
        
    Example:
        >>> bands = load_satellite_bands('data/2023_05_15/')
        >>> print(bands['B04'].shape)  # Red band
        (1024, 1024)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for loading GeoTIFF files. "
                         "Install with: pip install rasterio")
    
    bands = {}
    for band in band_names:
        filepath = os.path.join(data_dir, file_pattern.format(band=band))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Band file not found: {filepath}")
            
        with rasterio.open(filepath) as src:
            bands[band] = src.read(1).astype(np.float32)
            
    return bands


def normalize_bands(
    bands: Dict[str, np.ndarray],
    method: str = 'minmax',
    clip_percentile: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Apply radiometric normalization to satellite bands.
    
    Args:
        bands: Dictionary of band arrays
        method: Normalization method ('minmax', 'zscore', 'percentile')
        clip_percentile: Percentile for outlier clipping (default: 2%)
        
    Returns:
        Dictionary of normalized band arrays
        
    Note:
        As described in Supplementary Material S1.2, atmospheric correction
        and radiometric normalization are applied before analysis.
    """
    normalized = {}
    
    for band_name, band_data in bands.items():
        data = band_data.copy()
        
        # Clip outliers
        if clip_percentile > 0:
            p_low = np.percentile(data[data > 0], clip_percentile)
            p_high = np.percentile(data[data > 0], 100 - clip_percentile)
            data = np.clip(data, p_low, p_high)
        
        if method == 'minmax':
            data_min = np.min(data[data > 0]) if np.any(data > 0) else 0
            data_max = np.max(data)
            if data_max > data_min:
                normalized[band_name] = (data - data_min) / (data_max - data_min)
            else:
                normalized[band_name] = data
                
        elif method == 'zscore':
            mean = np.mean(data[data > 0]) if np.any(data > 0) else 0
            std = np.std(data[data > 0]) if np.any(data > 0) else 1
            normalized[band_name] = (data - mean) / (std + 1e-10)
            
        elif method == 'percentile':
            p1, p99 = np.percentile(data[data > 0], [1, 99])
            normalized[band_name] = np.clip((data - p1) / (p99 - p1 + 1e-10), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    return normalized


def stack_bands(
    bands: Dict[str, np.ndarray],
    band_order: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
) -> np.ndarray:
    """
    Stack multiple bands into a single multi-channel array.
    
    Args:
        bands: Dictionary of band arrays
        band_order: Order of bands in the stack (default: model input order)
        
    Returns:
        3D array of shape (height, width, n_bands)
        
    Note:
        The default band order matches the CNN model input:
        [Blue, Green, Red, NIR, SWIR1, SWIR2]
    """
    arrays = []
    for band_name in band_order:
        if band_name not in bands:
            raise KeyError(f"Band {band_name} not found in bands dictionary")
        arrays.append(bands[band_name])
        
    return np.stack(arrays, axis=-1)


def load_training_patches(
    patches_dir: str,
    patch_size: Tuple[int, int] = (256, 256),
    n_bands: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted training patches from directory.
    
    Args:
        patches_dir: Directory containing patch files
        patch_size: Expected patch dimensions (default: 256x256 as per manuscript)
        n_bands: Number of spectral bands (default: 6)
        
    Returns:
        Tuple of (patches, labels) arrays
        
    Note:
        Patches should be stored as .npy files with corresponding _label.npy files
        
    Example:
        >>> X, y = load_training_patches('data/patches/')
        >>> print(X.shape)  # (2500, 256, 256, 6)
        >>> print(y.shape)  # (2500,) or (2500, 256, 256)
    """
    patches = []
    labels = []
    
    patch_files = sorted([f for f in os.listdir(patches_dir) 
                          if f.endswith('.npy') and not f.endswith('_label.npy')])
    
    for patch_file in patch_files:
        patch_path = os.path.join(patches_dir, patch_file)
        label_path = os.path.join(patches_dir, patch_file.replace('.npy', '_label.npy'))
        
        patch = np.load(patch_path)
        
        if patch.shape[:2] != patch_size:
            warnings.warn(f"Patch {patch_file} has unexpected size {patch.shape[:2]}")
            
        patches.append(patch)
        
        if os.path.exists(label_path):
            labels.append(np.load(label_path))
        else:
            warnings.warn(f"Label file not found for {patch_file}")
            
    X = np.array(patches, dtype=np.float32)
    y = np.array(labels)
    
    return X, y


def create_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    validation_split: float = 0.0
) -> Union['tf.data.Dataset', Tuple['tf.data.Dataset', 'tf.data.Dataset']]:
    """
    Create TensorFlow dataset for model training.
    
    Args:
        X: Input patches array (N, H, W, C)
        y: Labels array
        batch_size: Training batch size (default: 32 as per manuscript)
        shuffle: Whether to shuffle data
        augment: Whether to apply data augmentation
        validation_split: Fraction for validation (0.0 means no split)
        
    Returns:
        TensorFlow Dataset object, or tuple of (train_ds, val_ds) if split > 0
        
    Note:
        Batch size of 32 and augmentation settings match manuscript specifications.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for dataset creation. "
                         "Install with: pip install tensorflow")
    
    # Ensure correct dtypes
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    if validation_split > 0:
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(X_train))
            
        if augment:
            train_ds = train_ds.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
        
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
            
        if augment:
            dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


def _augment_fn(image, label):
    """
    Apply augmentation to a single sample.
    
    Augmentation settings match manuscript (SI S1.4):
    - Random rotations (0°, 90°, 180°, 270°)
    - Horizontal and vertical flips
    - Brightness adjustments (±15%)
    - Gaussian noise (σ=0.01)
    """
    if not TF_AVAILABLE:
        return image, label
        
    # Random rotation (k * 90 degrees)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    if len(label.shape) > 0:  # If label is also an image
        label = tf.image.rot90(label, k)
    
    # Random flips
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        if len(label.shape) > 0:
            label = tf.image.flip_left_right(label)
            
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
        if len(label.shape) > 0:
            label = tf.image.flip_up_down(label)
    
    # Brightness adjustment (±15%)
    brightness_delta = tf.random.uniform([], -0.15, 0.15)
    image = image + brightness_delta
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Gaussian noise (σ=0.01)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def load_time_series(
    data_dir: str,
    date_format: str = '%Y_%m_%d',
    band_names: List[str] = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load a time series of satellite images.
    
    Args:
        data_dir: Base directory containing date-named subdirectories
        date_format: Format of date in directory names
        band_names: Bands to load for each date
        
    Returns:
        Tuple of (list of stacked images, list of date strings)
        
    Example:
        >>> images, dates = load_time_series('data/')
        >>> print(len(images))  # Number of dates
        >>> print(images[0].shape)  # (height, width, 6)
    """
    import datetime
    
    date_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            try:
                datetime.datetime.strptime(item, date_format)
                date_dirs.append(item)
            except ValueError:
                continue
                
    date_dirs = sorted(date_dirs)
    
    images = []
    dates = []
    
    for date_str in date_dirs:
        date_path = os.path.join(data_dir, date_str)
        try:
            bands = load_satellite_bands(date_path, band_names)
            bands = normalize_bands(bands)
            stacked = stack_bands(bands)
            images.append(stacked)
            dates.append(date_str)
        except Exception as e:
            warnings.warn(f"Could not load data for {date_str}: {e}")
            
    return images, dates


if __name__ == '__main__':
    # Example usage and testing
    print("Data Loader Module")
    print("=" * 50)
    
    print("\nLandsat-8 Band Specifications:")
    for band, info in LANDSAT8_BANDS.items():
        print(f"  {band}: {info['name']} ({info['wavelength']} nm, {info['resolution']}m)")
    
    # Create synthetic test data
    print("\nCreating synthetic test data...")
    synthetic_bands = {
        'B02': np.random.rand(256, 256).astype(np.float32) * 0.3,
        'B03': np.random.rand(256, 256).astype(np.float32) * 0.3,
        'B04': np.random.rand(256, 256).astype(np.float32) * 0.3,
        'B08': np.random.rand(256, 256).astype(np.float32) * 0.5,
        'B11': np.random.rand(256, 256).astype(np.float32) * 0.4,
        'B12': np.random.rand(256, 256).astype(np.float32) * 0.3,
    }
    
    # Test normalization
    normalized = normalize_bands(synthetic_bands, method='minmax')
    print(f"Normalized B04 range: [{normalized['B04'].min():.3f}, {normalized['B04'].max():.3f}]")
    
    # Test stacking
    stacked = stack_bands(normalized)
    print(f"Stacked array shape: {stacked.shape}")
    
    # Test dataset creation if TensorFlow available
    if TF_AVAILABLE:
        X = np.random.rand(100, 256, 256, 6).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        
        train_ds, val_ds = create_dataset(X, y, batch_size=32, 
                                          validation_split=0.2, augment=True)
        print(f"\nDataset created successfully")
        
        for batch_x, batch_y in train_ds.take(1):
            print(f"Batch X shape: {batch_x.shape}")
            print(f"Batch y shape: {batch_y.shape}")
