"""
Patch extraction utilities for satellite imagery.

This module provides functions for extracting training patches from
full satellite scenes for the CNN model training.

Patch specifications from manuscript:
- Size: 256×256 pixels (as per SI S1.4)
- Original count: 2,500 patches
- Field-validated labels

References:
    Supplementary Material S1.4: Training Dataset Construction
"""

import numpy as np
from typing import Tuple, List, Optional, Generator
import warnings


def extract_patches(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (256, 256),
    stride: int = None,
    padding: str = 'valid'
) -> np.ndarray:
    """
    Extract non-overlapping or overlapping patches from an image.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        patch_size: Size of patches (default: 256×256 as per manuscript)
        stride: Step between patches (default: patch_size for non-overlapping)
        padding: 'valid' (no padding) or 'same' (pad to include edges)
        
    Returns:
        Array of patches (N, pH, pW, C) or (N, pH, pW)
        
    Example:
        >>> image = np.random.rand(1024, 1024, 6)
        >>> patches = extract_patches(image, patch_size=(256, 256))
        >>> print(patches.shape)  # (16, 256, 256, 6)
    """
    if stride is None:
        stride = patch_size[0]
    
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    # Handle padding
    if padding == 'same':
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        
        if len(image.shape) == 3:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        h, w = image.shape[:2]
    
    # Calculate number of patches
    n_rows = (h - ph) // stride + 1
    n_cols = (w - pw) // stride + 1
    
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * stride
            x_start = j * stride
            patch = image[y_start:y_start+ph, x_start:x_start+pw]
            patches.append(patch)
    
    return np.array(patches)


def extract_patches_with_labels(
    image: np.ndarray,
    label_mask: np.ndarray,
    patch_size: Tuple[int, int] = (256, 256),
    stride: int = None,
    min_label_fraction: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patches with corresponding labels.
    
    Args:
        image: Input image (H, W, C)
        label_mask: Binary label mask (H, W)
        patch_size: Size of patches
        stride: Step between patches
        min_label_fraction: Minimum fraction of labeled pixels in patch
        
    Returns:
        Tuple of (image_patches, label_patches)
    """
    if stride is None:
        stride = patch_size[0]
    
    image_patches = extract_patches(image, patch_size, stride)
    label_patches = extract_patches(label_mask, patch_size, stride)
    
    # Filter by minimum label fraction
    if min_label_fraction > 0:
        valid_mask = []
        for label_patch in label_patches:
            fraction = np.mean(label_patch > 0)
            valid_mask.append(fraction >= min_label_fraction)
        
        valid_mask = np.array(valid_mask)
        image_patches = image_patches[valid_mask]
        label_patches = label_patches[valid_mask]
    
    return image_patches, label_patches


def create_patch_dataset(
    images: List[np.ndarray],
    labels: List[np.ndarray],
    patch_size: Tuple[int, int] = (256, 256),
    n_patches_per_image: int = None,
    random_sampling: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of patches from multiple images.
    
    Args:
        images: List of input images
        labels: List of label masks
        patch_size: Size of patches
        n_patches_per_image: Number of patches to extract per image
        random_sampling: Use random sampling vs systematic grid
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) arrays
        
    Note:
        This function is used to create the 2,500 patch training set
        described in the manuscript.
    """
    np.random.seed(seed)
    
    all_patches = []
    all_labels = []
    
    for img, lbl in zip(images, labels):
        h, w = img.shape[:2]
        ph, pw = patch_size
        
        if random_sampling and n_patches_per_image:
            # Random patch locations
            for _ in range(n_patches_per_image):
                y = np.random.randint(0, h - ph + 1)
                x = np.random.randint(0, w - pw + 1)
                
                patch = img[y:y+ph, x:x+pw]
                label = lbl[y:y+ph, x:x+pw]
                
                all_patches.append(patch)
                all_labels.append(label)
        else:
            # Systematic extraction
            patches, labels_p = extract_patches_with_labels(img, lbl, patch_size)
            
            if n_patches_per_image and len(patches) > n_patches_per_image:
                indices = np.random.choice(len(patches), n_patches_per_image, replace=False)
                patches = patches[indices]
                labels_p = labels_p[indices]
            
            all_patches.extend(patches)
            all_labels.extend(labels_p)
    
    return np.array(all_patches), np.array(all_labels)


def stratified_patch_sampling(
    image: np.ndarray,
    label_mask: np.ndarray,
    patch_size: Tuple[int, int] = (256, 256),
    n_patches: int = 100,
    class_ratios: dict = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patches with stratified sampling by class.
    
    Args:
        image: Input image
        label_mask: Class labels (can be multi-class)
        patch_size: Size of patches
        n_patches: Total number of patches to extract
        class_ratios: Dict of {class: ratio} (default: proportional)
        seed: Random seed
        
    Returns:
        Tuple of (image_patches, label_patches)
        
    Note:
        Useful for handling class imbalance in contaminated/non-contaminated areas.
    """
    np.random.seed(seed)
    
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    # Get unique classes
    classes = np.unique(label_mask)
    
    if class_ratios is None:
        # Proportional sampling
        class_ratios = {}
        for cls in classes:
            class_ratios[cls] = np.mean(label_mask == cls)
    
    patches = []
    labels = []
    
    for cls, ratio in class_ratios.items():
        n_class_patches = int(n_patches * ratio)
        
        # Find valid patch locations for this class
        class_mask = label_mask == cls
        
        valid_patches = []
        valid_labels = []
        
        attempts = 0
        max_attempts = n_class_patches * 10
        
        while len(valid_patches) < n_class_patches and attempts < max_attempts:
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)
            
            patch_label = label_mask[y:y+ph, x:x+pw]
            
            # Check if patch is majority this class
            if np.mean(patch_label == cls) >= 0.5:
                valid_patches.append(image[y:y+ph, x:x+pw])
                valid_labels.append(patch_label)
            
            attempts += 1
        
        patches.extend(valid_patches)
        labels.extend(valid_labels)
    
    # Shuffle
    indices = np.random.permutation(len(patches))
    patches = [patches[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return np.array(patches), np.array(labels)


def patch_generator(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (256, 256),
    stride: int = None,
    batch_size: int = 32
) -> Generator[np.ndarray, None, None]:
    """
    Generate patches in batches for memory-efficient processing.
    
    Args:
        image: Input image
        patch_size: Size of patches
        stride: Step between patches
        batch_size: Number of patches per batch
        
    Yields:
        Batches of patches
        
    Example:
        >>> for batch in patch_generator(large_image, batch_size=32):
        ...     predictions = model.predict(batch)
    """
    if stride is None:
        stride = patch_size[0]
    
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    n_rows = (h - ph) // stride + 1
    n_cols = (w - pw) // stride + 1
    
    batch = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * stride
            x_start = j * stride
            patch = image[y_start:y_start+ph, x_start:x_start+pw]
            batch.append(patch)
            
            if len(batch) == batch_size:
                yield np.array(batch)
                batch = []
    
    if batch:  # Remaining patches
        yield np.array(batch)


def reconstruct_from_patches(
    patches: np.ndarray,
    image_shape: Tuple[int, ...],
    patch_size: Tuple[int, int] = (256, 256),
    stride: int = None,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Reconstruct full image from patches.
    
    Args:
        patches: Array of patches (N, pH, pW, C) or (N, pH, pW)
        image_shape: Shape of original image
        patch_size: Size of patches
        stride: Stride used during extraction
        aggregation: How to handle overlapping regions ('mean', 'max', 'last')
        
    Returns:
        Reconstructed image
        
    Note:
        Useful for creating prediction maps from patch-based predictions.
    """
    if stride is None:
        stride = patch_size[0]
    
    h, w = image_shape[:2]
    ph, pw = patch_size
    
    has_channels = len(patches.shape) == 4
    if has_channels:
        channels = patches.shape[-1]
        reconstructed = np.zeros((h, w, channels), dtype=np.float32)
    else:
        reconstructed = np.zeros((h, w), dtype=np.float32)
    
    counts = np.zeros((h, w), dtype=np.float32)
    
    n_rows = (h - ph) // stride + 1
    n_cols = (w - pw) // stride + 1
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            y_start = i * stride
            x_start = j * stride
            
            if aggregation == 'mean':
                reconstructed[y_start:y_start+ph, x_start:x_start+pw] += patches[idx]
                counts[y_start:y_start+ph, x_start:x_start+pw] += 1
            elif aggregation == 'max':
                if has_channels:
                    current = reconstructed[y_start:y_start+ph, x_start:x_start+pw]
                    reconstructed[y_start:y_start+ph, x_start:x_start+pw] = np.maximum(current, patches[idx])
                else:
                    current = reconstructed[y_start:y_start+ph, x_start:x_start+pw]
                    reconstructed[y_start:y_start+ph, x_start:x_start+pw] = np.maximum(current, patches[idx])
            else:  # last
                reconstructed[y_start:y_start+ph, x_start:x_start+pw] = patches[idx]
            
            idx += 1
    
    if aggregation == 'mean':
        counts = np.maximum(counts, 1)
        if has_channels:
            reconstructed /= counts[:, :, np.newaxis]
        else:
            reconstructed /= counts
    
    return reconstructed


if __name__ == '__main__':
    print("Patch Extraction Module")
    print("=" * 50)
    
    # Create sample image
    image = np.random.rand(1024, 1024, 6).astype(np.float32)
    label = (np.random.rand(1024, 1024) > 0.7).astype(np.int32)
    
    print(f"\nOriginal image shape: {image.shape}")
    print(f"Label mask shape: {label.shape}")
    
    # Extract patches (256×256 as per manuscript)
    patches = extract_patches(image, patch_size=(256, 256))
    print(f"\nExtracted patches: {patches.shape}")
    print(f"Expected: 16 patches (4×4 grid from 1024×1024)")
    
    # Extract with labels
    img_patches, lbl_patches = extract_patches_with_labels(
        image, label, patch_size=(256, 256)
    )
    print(f"\nPatches with labels: {img_patches.shape}, {lbl_patches.shape}")
    
    # Stratified sampling
    strat_patches, strat_labels = stratified_patch_sampling(
        image, label, patch_size=(256, 256), n_patches=50
    )
    print(f"\nStratified sampled: {strat_patches.shape}")
    
    # Test reconstruction
    reconstructed = reconstruct_from_patches(
        patches, image.shape, patch_size=(256, 256)
    )
    print(f"\nReconstructed image shape: {reconstructed.shape}")
    
    # Verify reconstruction
    diff = np.abs(image - reconstructed).max()
    print(f"Max reconstruction difference: {diff:.6f}")
    
    print("\n" + "=" * 50)
    print("Manuscript specifications:")
    print("  - Patch size: 256×256 pixels ✓")
    print("  - Original patches: 2,500")
    print("  - After augmentation: ~20,000 samples")
