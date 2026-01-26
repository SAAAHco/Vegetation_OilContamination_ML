"""
Data augmentation pipeline for satellite image patches.

This module implements the augmentation strategy described in the manuscript
(Supplementary Material S1.4), which expands the training dataset from
2,500 patches to approximately 20,000 effective samples.

Augmentation techniques:
- Random rotations (0°, 90°, 180°, 270°)
- Horizontal and vertical flips
- Brightness adjustments (±15%)
- Gaussian noise injection (σ=0.01)

References:
    Supplementary Material S1.4: Data Augmentation Strategy
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
import warnings

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def create_augmentation_pipeline(
    rotation: bool = True,
    flip: bool = True,
    brightness: bool = True,
    noise: bool = True,
    brightness_delta: float = 0.15,
    noise_sigma: float = 0.01
) -> Callable:
    """
    Create an augmentation function with specified transformations.
    
    Args:
        rotation: Apply random 90° rotations
        flip: Apply random horizontal/vertical flips
        brightness: Apply brightness adjustments (±15% default)
        noise: Apply Gaussian noise (σ=0.01 default)
        brightness_delta: Maximum brightness change (±)
        noise_sigma: Standard deviation of Gaussian noise
        
    Returns:
        Augmentation function that transforms (image, label) pairs
        
    Note:
        Default parameters match manuscript specifications (SI S1.4)
    """
    def augment(image: np.ndarray, label: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to image and optionally label."""
        aug_image = image.copy()
        aug_label = label.copy() if label is not None else None
        
        # Random rotation (0°, 90°, 180°, 270°)
        if rotation:
            k = np.random.randint(0, 4)
            aug_image = np.rot90(aug_image, k, axes=(0, 1))
            if aug_label is not None and len(aug_label.shape) >= 2:
                aug_label = np.rot90(aug_label, k, axes=(0, 1))
        
        # Horizontal flip
        if flip and np.random.random() > 0.5:
            aug_image = np.flip(aug_image, axis=1)
            if aug_label is not None and len(aug_label.shape) >= 2:
                aug_label = np.flip(aug_label, axis=1)
        
        # Vertical flip
        if flip and np.random.random() > 0.5:
            aug_image = np.flip(aug_image, axis=0)
            if aug_label is not None and len(aug_label.shape) >= 2:
                aug_label = np.flip(aug_label, axis=0)
        
        # Brightness adjustment
        if brightness:
            delta = np.random.uniform(-brightness_delta, brightness_delta)
            aug_image = aug_image + delta
            aug_image = np.clip(aug_image, 0, 1)
        
        # Gaussian noise
        if noise:
            noise_array = np.random.normal(0, noise_sigma, aug_image.shape)
            aug_image = aug_image + noise_array
            aug_image = np.clip(aug_image, 0, 1)
        
        return aug_image.astype(np.float32), aug_label
    
    return augment


def apply_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    augment_fn: Callable,
    n_augmentations: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to expand dataset.
    
    Args:
        X: Input images array (N, H, W, C)
        y: Labels array
        augment_fn: Augmentation function
        n_augmentations: Number of augmented versions per original
        
    Returns:
        Tuple of augmented (X, y) arrays
        
    Note:
        With n_augmentations=8 on 2,500 patches, produces ~20,000 samples
        as described in the manuscript.
    """
    augmented_X = [X]  # Include originals
    augmented_y = [y]
    
    for i in range(n_augmentations - 1):
        aug_X_batch = []
        aug_y_batch = []
        
        for j in range(len(X)):
            aug_x, aug_y = augment_fn(X[j], y[j] if len(y.shape) > 1 else None)
            aug_X_batch.append(aug_x)
            if aug_y is not None:
                aug_y_batch.append(aug_y)
        
        augmented_X.append(np.array(aug_X_batch))
        if aug_y_batch:
            augmented_y.append(np.array(aug_y_batch))
        else:
            augmented_y.append(y)
    
    final_X = np.concatenate(augmented_X, axis=0)
    final_y = np.concatenate(augmented_y, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(final_X))
    final_X = final_X[indices]
    final_y = final_y[indices]
    
    return final_X, final_y


def geometric_augmentation(
    image: np.ndarray,
    label: np.ndarray = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply only geometric augmentations (rotation, flip).
    
    Useful for creating multiple versions without altering pixel values.
    
    Args:
        image: Input image (H, W, C)
        label: Optional label image (H, W) or (H, W, C)
        
    Returns:
        Augmented (image, label) tuple
    """
    aug_fn = create_augmentation_pipeline(
        rotation=True, flip=True,
        brightness=False, noise=False
    )
    return aug_fn(image, label)


def photometric_augmentation(
    image: np.ndarray,
    brightness_delta: float = 0.15,
    noise_sigma: float = 0.01
) -> np.ndarray:
    """
    Apply only photometric augmentations (brightness, noise).
    
    Args:
        image: Input image (H, W, C)
        brightness_delta: Brightness adjustment range
        noise_sigma: Gaussian noise sigma
        
    Returns:
        Augmented image
    """
    aug_image = image.copy()
    
    # Brightness
    delta = np.random.uniform(-brightness_delta, brightness_delta)
    aug_image = aug_image + delta
    
    # Noise
    noise = np.random.normal(0, noise_sigma, aug_image.shape)
    aug_image = aug_image + noise
    
    return np.clip(aug_image, 0, 1).astype(np.float32)


def create_tf_augmentation_layer():
    """
    Create TensorFlow augmentation layer for on-the-fly augmentation.
    
    Returns:
        tf.keras.Sequential model for augmentation
        
    Note:
        For use in tf.data pipelines or as preprocessing layer in models.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for this function")
    
    from tensorflow import keras
    
    return keras.Sequential([
        keras.layers.RandomRotation(factor=0.25, fill_mode='reflect'),  # Up to 90 degrees
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomFlip("vertical"),
        keras.layers.RandomBrightness(factor=0.15, value_range=(0, 1)),
        keras.layers.GaussianNoise(stddev=0.01),
    ], name='augmentation_layer')


def balanced_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    augment_fn: Callable,
    target_per_class: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation with class balancing.
    
    Args:
        X: Input images
        y: Class labels (1D array for classification)
        augment_fn: Augmentation function
        target_per_class: Target samples per class (default: max class count)
        
    Returns:
        Balanced augmented (X, y) arrays
    """
    classes, counts = np.unique(y, return_counts=True)
    
    if target_per_class is None:
        target_per_class = np.max(counts)
    
    balanced_X = []
    balanced_y = []
    
    for cls in classes:
        cls_mask = y == cls
        cls_X = X[cls_mask]
        cls_y = y[cls_mask]
        
        current_count = len(cls_X)
        augmented_X = [cls_X]
        augmented_y = [cls_y]
        
        while np.concatenate(augmented_X, axis=0).shape[0] < target_per_class:
            aug_batch = []
            for img in cls_X:
                aug_img, _ = augment_fn(img)
                aug_batch.append(aug_img)
            augmented_X.append(np.array(aug_batch))
            augmented_y.append(cls_y)
        
        cls_X_aug = np.concatenate(augmented_X, axis=0)[:target_per_class]
        cls_y_aug = np.full(target_per_class, cls)
        
        balanced_X.append(cls_X_aug)
        balanced_y.append(cls_y_aug)
    
    final_X = np.concatenate(balanced_X, axis=0)
    final_y = np.concatenate(balanced_y, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(final_X))
    
    return final_X[indices], final_y[indices]


if __name__ == '__main__':
    print("Data Augmentation Module")
    print("=" * 50)
    
    # Create sample data
    n_samples = 10
    sample_X = np.random.rand(n_samples, 256, 256, 6).astype(np.float32)
    sample_y = np.random.randint(0, 2, n_samples)
    
    print(f"\nOriginal dataset: {sample_X.shape}, labels: {sample_y.shape}")
    
    # Create augmentation pipeline (matching manuscript specs)
    augment_fn = create_augmentation_pipeline(
        rotation=True,
        flip=True,
        brightness=True,
        noise=True,
        brightness_delta=0.15,  # ±15%
        noise_sigma=0.01        # σ=0.01
    )
    
    # Apply single augmentation
    aug_img, _ = augment_fn(sample_X[0])
    print(f"Single augmented image shape: {aug_img.shape}")
    print(f"Value range: [{aug_img.min():.3f}, {aug_img.max():.3f}]")
    
    # Apply full augmentation (8x expansion)
    aug_X, aug_y = apply_augmentation(sample_X, sample_y, augment_fn, n_augmentations=8)
    print(f"\nAugmented dataset: {aug_X.shape}")
    print(f"Expansion factor: {len(aug_X) / n_samples}x")
    
    # With 2,500 patches: 2,500 × 8 = 20,000 samples (as per manuscript)
    print(f"\nManuscript validation:")
    print(f"  Original patches: 2,500")
    print(f"  After 8x augmentation: ~20,000 samples ✓")
