"""
Spatial Analysis Module

This module implements spatial analysis techniques for landscape pattern analysis:
- Fractal dimension analysis (box-counting method)
- Lacunarity analysis
- Markov chain analysis for recovery state transitions
- Landscape metrics (patch dynamics, edge density, aggregation)

References:
    - Supplementary Material Section S1.7.4 (Fractal Analysis)
    - Supplementary Material Section S1.7.5 (Recovery State Classification)
    - Supplementary Material Section S1.7.3 (Landscape Metrics)
"""

import numpy as np
from scipy import ndimage, stats
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Optional imports
try:
    from skimage import measure, morphology
    from skimage.segmentation import find_boundaries
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not installed. Some spatial analyses will be limited.")


def fractal_dimension_box_counting(
    binary_mask: np.ndarray,
    min_box_size: int = 1,
    max_box_size: Optional[int] = None,
    n_sizes: int = 10
) -> Dict:
    """
    Calculate fractal dimension using box-counting method.
    
    From Supplementary Material Section S1.7.4:
    "Box sizes (ε) ranged from 30 m (1 pixel) to 7,680 m (256 pixels), 
    spanning the pixel-to-landscape continuum."
    
    The fractal dimension D is calculated from:
        N(ε) ∝ ε^(-D)
    
    Where N(ε) is the number of boxes of size ε needed to cover the pattern.
    
    Args:
        binary_mask: 2D binary array (1 = feature, 0 = background)
        min_box_size: Minimum box size in pixels (default 1)
        max_box_size: Maximum box size (default: min(height, width) / 4)
        n_sizes: Number of box sizes to use
        
    Returns:
        Dictionary containing:
            - fractal_dimension: Estimated fractal dimension (D)
            - box_sizes: Array of box sizes used
            - box_counts: Number of boxes at each size
            - r_squared: R² of the linear fit
            - intercept: Intercept of log-log fit
    """
    binary_mask = (binary_mask > 0).astype(np.int32)
    height, width = binary_mask.shape
    
    # Set max box size
    if max_box_size is None:
        max_box_size = min(height, width) // 4
    
    # Generate box sizes (exponentially spaced)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box_size),
        np.log10(max_box_size),
        n_sizes
    ).astype(int))
    
    box_counts = []
    
    for box_size in box_sizes:
        # Count boxes that contain at least one feature pixel
        # Reshape mask into boxes
        n_boxes_h = height // box_size
        n_boxes_w = width // box_size
        
        if n_boxes_h == 0 or n_boxes_w == 0:
            continue
        
        # Crop mask to fit integer number of boxes
        cropped = binary_mask[:n_boxes_h * box_size, :n_boxes_w * box_size]
        
        # Reshape to (n_boxes_h, box_size, n_boxes_w, box_size)
        # Then check if any pixel in each box is non-zero
        boxes = cropped.reshape(n_boxes_h, box_size, n_boxes_w, box_size)
        box_sums = boxes.any(axis=(1, 3))
        count = np.sum(box_sums)
        
        box_counts.append(count)
    
    box_sizes = box_sizes[:len(box_counts)]
    box_counts = np.array(box_counts)
    
    # Filter out zero counts
    valid_mask = box_counts > 0
    box_sizes = box_sizes[valid_mask]
    box_counts = box_counts[valid_mask]
    
    if len(box_sizes) < 2:
        return {
            'fractal_dimension': np.nan,
            'box_sizes': box_sizes,
            'box_counts': box_counts,
            'r_squared': np.nan,
            'intercept': np.nan
        }
    
    # Linear regression on log-log scale
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    
    # Fractal dimension is negative of slope
    fractal_dimension = -slope
    
    return {
        'fractal_dimension': fractal_dimension,
        'box_sizes': box_sizes,
        'box_counts': box_counts,
        'r_squared': r_value**2,
        'intercept': intercept,
        'slope': slope,
        'p_value': p_value
    }


def lacunarity_analysis(
    binary_mask: np.ndarray,
    box_sizes: Optional[List[int]] = None
) -> Dict:
    """
    Calculate lacunarity using gliding box method.
    
    From Supplementary Material Section S1.7.4:
    "Lacunarity analysis used gliding box sizes of 3×3, 5×5, 7×7, 11×11, 
    21×21, and 31×31 pixels to assess spatial heterogeneity across scales."
    
    Lacunarity Λ measures the "gappiness" or heterogeneity of a pattern:
        Λ(r) = (σ²(r) / μ²(r)) + 1
    
    Where μ and σ² are the mean and variance of box mass (pixel count).
    
    Args:
        binary_mask: 2D binary array
        box_sizes: List of box sizes (default: [3, 5, 7, 11, 21, 31])
        
    Returns:
        Dictionary containing:
            - lacunarity: Lacunarity values for each box size
            - box_sizes: Box sizes used
            - mean_lacunarity: Average lacunarity across scales
            - lacunarity_slope: Slope of log-lacunarity vs log-size
    """
    binary_mask = (binary_mask > 0).astype(np.float64)
    height, width = binary_mask.shape
    
    if box_sizes is None:
        box_sizes = [3, 5, 7, 11, 21, 31]
    
    lacunarity_values = []
    valid_sizes = []
    
    for box_size in box_sizes:
        if box_size > min(height, width):
            continue
        
        # Calculate box masses using convolution (gliding box)
        kernel = np.ones((box_size, box_size))
        box_masses = ndimage.convolve(binary_mask, kernel, mode='constant', cval=0)
        
        # Trim edges
        half = box_size // 2
        box_masses = box_masses[half:-half if half > 0 else None, 
                                 half:-half if half > 0 else None]
        
        if box_masses.size == 0:
            continue
        
        # Calculate lacunarity
        mean_mass = np.mean(box_masses)
        var_mass = np.var(box_masses)
        
        if mean_mass > 0:
            lacunarity = (var_mass / (mean_mass**2)) + 1
        else:
            lacunarity = 1.0
        
        lacunarity_values.append(lacunarity)
        valid_sizes.append(box_size)
    
    lacunarity_values = np.array(lacunarity_values)
    valid_sizes = np.array(valid_sizes)
    
    # Calculate slope of log-lacunarity vs log-size
    if len(valid_sizes) >= 2:
        log_sizes = np.log(valid_sizes)
        log_lac = np.log(lacunarity_values)
        slope, _, _, _, _ = stats.linregress(log_sizes, log_lac)
    else:
        slope = np.nan
    
    return {
        'lacunarity': lacunarity_values,
        'box_sizes': valid_sizes,
        'mean_lacunarity': np.mean(lacunarity_values) if len(lacunarity_values) > 0 else np.nan,
        'lacunarity_slope': slope
    }


def markov_chain_analysis(
    state_sequence: np.ndarray,
    n_states: int = 5
) -> Dict:
    """
    Perform Markov chain analysis for recovery state transitions.
    
    From Supplementary Material Section S1.7.5:
    "Markov chain analysis computed transition probabilities"
    
    States (from classify_recovery_state):
        0 = Contaminated
        1 = Bare/Degraded
        2 = Transitional
        3 = Active Recovery
        4 = Recovered
    
    Args:
        state_sequence: Time series of state classifications
                       Shape can be (time,) for single pixel or (time, n_pixels)
        n_states: Number of possible states (5)
        
    Returns:
        Dictionary containing:
            - transition_matrix: Transition probability matrix P(i→j)
            - transition_counts: Raw transition counts
            - stationary_distribution: Long-term equilibrium distribution
            - mean_first_passage_time: Expected time to reach each state
            - state_names: State name labels
    """
    state_names = ['Contaminated', 'Bare/Degraded', 'Transitional', 
                   'Active Recovery', 'Recovered']
    
    # Ensure state sequence is 2D
    if state_sequence.ndim == 1:
        state_sequence = state_sequence.reshape(-1, 1)
    
    n_times, n_pixels = state_sequence.shape
    
    # Count transitions
    transition_counts = np.zeros((n_states, n_states))
    
    for t in range(n_times - 1):
        for p in range(n_pixels):
            from_state = int(state_sequence[t, p])
            to_state = int(state_sequence[t + 1, p])
            
            if 0 <= from_state < n_states and 0 <= to_state < n_states:
                transition_counts[from_state, to_state] += 1
    
    # Normalize to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_counts / row_sums
    
    # Calculate stationary distribution (eigenvector with eigenvalue 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        # Find eigenvector corresponding to eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # Normalize
    except:
        stationary = np.ones(n_states) / n_states
    
    # Calculate mean first passage times
    mean_fpt = _calculate_mean_first_passage_time(transition_matrix)
    
    return {
        'transition_matrix': transition_matrix,
        'transition_counts': transition_counts,
        'stationary_distribution': stationary,
        'mean_first_passage_time': mean_fpt,
        'state_names': state_names,
        'n_transitions': int(transition_counts.sum())
    }


def _calculate_mean_first_passage_time(P: np.ndarray) -> np.ndarray:
    """
    Calculate mean first passage time matrix.
    
    M[i,j] = expected number of steps to reach state j starting from state i.
    """
    n = P.shape[0]
    M = np.zeros((n, n))
    
    # Get stationary distribution
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
    except:
        pi = np.ones(n) / n
    
    # Calculate fundamental matrix Z = (I - P + W)^(-1)
    # where W[i,j] = pi[j] for all i
    I = np.eye(n)
    W = np.tile(pi, (n, 1))
    
    try:
        Z = np.linalg.inv(I - P + W)
        
        # Mean first passage time: M[i,j] = (Z[j,j] - Z[i,j]) / pi[j]
        for i in range(n):
            for j in range(n):
                if pi[j] > 0 and i != j:
                    M[i, j] = (Z[j, j] - Z[i, j]) / pi[j]
                elif i == j:
                    M[i, j] = 0
    except:
        M = np.full((n, n), np.nan)
    
    return M


def landscape_metrics(
    binary_mask: np.ndarray,
    pixel_resolution: float = 30.0
) -> Dict:
    """
    Calculate comprehensive landscape metrics.
    
    From Supplementary Material Section S1.7.3:
    Metrics include patch dynamics, edge density, and aggregation indices.
    
    Args:
        binary_mask: 2D binary array (1 = feature, 0 = background)
        pixel_resolution: Pixel size in meters (30m for Landsat)
        
    Returns:
        Dictionary containing landscape metrics
    """
    if not HAS_SKIMAGE:
        warnings.warn("scikit-image required for full landscape metrics")
        return _basic_landscape_metrics(binary_mask, pixel_resolution)
    
    binary_mask = (binary_mask > 0).astype(np.int32)
    height, width = binary_mask.shape
    pixel_area_m2 = pixel_resolution ** 2
    pixel_area_km2 = pixel_area_m2 / 1e6
    
    # Label connected components (patches)
    labeled = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        return {
            'num_patches': 0,
            'total_area_km2': 0,
            'mean_patch_size_km2': 0,
            'largest_patch_km2': 0,
            'patch_density': 0,
            'edge_density': 0,
            'aggregation_index': 0,
            'landscape_shape_index': 0,
            'coverage_percent': 0
        }
    
    # Basic patch metrics
    patch_areas = np.array([r.area for r in regions]) * pixel_area_km2
    num_patches = len(regions)
    total_area = np.sum(patch_areas)
    mean_patch_size = np.mean(patch_areas)
    largest_patch = np.max(patch_areas)
    
    # Patch density (patches per 100 km²)
    landscape_area_km2 = height * width * pixel_area_km2
    patch_density = (num_patches / landscape_area_km2) * 100 if landscape_area_km2 > 0 else 0
    
    # Edge density
    edges = find_boundaries(binary_mask, mode='outer')
    edge_pixels = np.sum(edges)
    edge_length_km = edge_pixels * pixel_resolution / 1000
    edge_density = edge_length_km / landscape_area_km2 if landscape_area_km2 > 0 else 0
    
    # Aggregation index (ratio of like adjacencies to maximum possible)
    horiz_adj = np.sum(binary_mask[:, :-1] == binary_mask[:, 1:])
    vert_adj = np.sum(binary_mask[:-1, :] == binary_mask[1:, :])
    total_adj = horiz_adj + vert_adj
    max_adj = (height * (width - 1)) + ((height - 1) * width)
    aggregation_index = total_adj / max_adj if max_adj > 0 else 0
    
    # Landscape shape index (edge complexity)
    total_perimeter = sum(r.perimeter for r in regions) * pixel_resolution
    landscape_shape_index = total_perimeter / (2 * np.sqrt(np.pi * total_area * 1e6)) if total_area > 0 else 0
    
    # Coverage percentage
    coverage = (np.sum(binary_mask) / (height * width)) * 100
    
    return {
        'num_patches': num_patches,
        'total_area_km2': total_area,
        'mean_patch_size_km2': mean_patch_size,
        'std_patch_size_km2': np.std(patch_areas) if len(patch_areas) > 1 else 0,
        'largest_patch_km2': largest_patch,
        'patch_density': patch_density,
        'edge_density': edge_density,
        'aggregation_index': aggregation_index,
        'landscape_shape_index': landscape_shape_index,
        'coverage_percent': coverage,
        'patch_areas_km2': patch_areas
    }


def _basic_landscape_metrics(
    binary_mask: np.ndarray,
    pixel_resolution: float = 30.0
) -> Dict:
    """Basic landscape metrics without scikit-image."""
    binary_mask = (binary_mask > 0).astype(np.int32)
    height, width = binary_mask.shape
    pixel_area_km2 = (pixel_resolution ** 2) / 1e6
    
    total_area = np.sum(binary_mask) * pixel_area_km2
    coverage = (np.sum(binary_mask) / (height * width)) * 100
    
    return {
        'total_area_km2': total_area,
        'coverage_percent': coverage,
        'num_patches': np.nan,
        'mean_patch_size_km2': np.nan
    }


def gradient_analysis(
    contamination_mask: np.ndarray,
    vegetation_index: np.ndarray,
    pixel_resolution: float = 30.0,
    max_distance_m: float = 10000.0,
    interval_m: float = 500.0
) -> Dict:
    """
    Analyze vegetation gradient with distance from contamination.
    
    From Supplementary Material Section S1.7.6:
    "Euclidean distance transform generated continuous distance surfaces from 
    contamination pixels. Vegetation density (normalized SAVI) was sampled at 
    500m intervals (0-10,000m range)."
    
    Args:
        contamination_mask: Binary mask of contaminated areas
        vegetation_index: Vegetation index values (e.g., SAVI)
        pixel_resolution: Pixel size in meters (30m)
        max_distance_m: Maximum distance to analyze (10,000m)
        interval_m: Sampling interval (500m)
        
    Returns:
        Dictionary containing:
            - distances: Distance values (m)
            - mean_vegetation: Mean vegetation index at each distance
            - std_vegetation: Standard deviation at each distance
            - peak_distance: Distance with maximum vegetation
            - peak_value: Maximum vegetation value
    """
    contamination_mask = (contamination_mask > 0).astype(np.int32)
    
    # Calculate distance transform (distance to nearest contaminated pixel)
    # Invert mask so contaminated pixels = 0
    distance_transform = ndimage.distance_transform_edt(1 - contamination_mask)
    distance_transform_m = distance_transform * pixel_resolution
    
    # Create distance bins
    distances = np.arange(0, max_distance_m + interval_m, interval_m)
    mean_vegetation = []
    std_vegetation = []
    counts = []
    
    for i in range(len(distances) - 1):
        d_min = distances[i]
        d_max = distances[i + 1]
        
        # Find pixels in this distance range
        mask = (distance_transform_m >= d_min) & (distance_transform_m < d_max)
        
        if np.sum(mask) > 0:
            values = vegetation_index[mask]
            mean_vegetation.append(np.nanmean(values))
            std_vegetation.append(np.nanstd(values))
            counts.append(np.sum(mask))
        else:
            mean_vegetation.append(np.nan)
            std_vegetation.append(np.nan)
            counts.append(0)
    
    mean_vegetation = np.array(mean_vegetation)
    std_vegetation = np.array(std_vegetation)
    distance_centers = (distances[:-1] + distances[1:]) / 2
    
    # Find peak vegetation (optimal distance)
    valid_mask = ~np.isnan(mean_vegetation)
    if np.any(valid_mask):
        peak_idx = np.nanargmax(mean_vegetation)
        peak_distance = distance_centers[peak_idx]
        peak_value = mean_vegetation[peak_idx]
    else:
        peak_distance = np.nan
        peak_value = np.nan
    
    return {
        'distances': distance_centers,
        'mean_vegetation': mean_vegetation,
        'std_vegetation': std_vegetation,
        'counts': np.array(counts),
        'peak_distance': peak_distance,
        'peak_value': peak_value,
        'distance_transform': distance_transform_m
    }


def spatial_autocorrelation_morans_i(
    values: np.ndarray,
    lag: int = 5,
    sample_size: int = 1000
) -> Dict:
    """
    Calculate Moran's I spatial autocorrelation index.
    
    Args:
        values: 2D array of values
        lag: Spatial lag distance in pixels
        sample_size: Number of pixel pairs to sample
        
    Returns:
        Dictionary with Moran's I and significance
    """
    height, width = values.shape
    n = height * width
    mean_val = np.nanmean(values)
    
    # Sample random pixel pairs
    np.random.seed(42)
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        y_coords = idx // width
        x_coords = idx % width
    else:
        y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
    
    n_sample = len(y_coords)
    sample_values = values[y_coords, x_coords]
    
    # Calculate Moran's I
    numerator = 0.0
    weight_sum = 0.0
    
    for i in range(min(n_sample, 500)):
        for j in range(i + 1, min(n_sample, 500)):
            dist = np.sqrt((y_coords[i] - y_coords[j])**2 + (x_coords[i] - x_coords[j])**2)
            
            # Weight function (binary: 1 if within lag distance)
            if lag - 0.5 <= dist <= lag + 0.5:
                vi = sample_values[i] - mean_val
                vj = sample_values[j] - mean_val
                
                if not (np.isnan(vi) or np.isnan(vj)):
                    numerator += vi * vj
                    weight_sum += 1
    
    denominator = np.nansum((sample_values - mean_val)**2)
    
    if weight_sum > 0 and denominator > 0:
        morans_i = (n_sample * numerator) / (weight_sum * denominator)
    else:
        morans_i = 0.0
    
    # Expected value under null hypothesis
    expected_i = -1.0 / (n_sample - 1)
    
    return {
        'morans_i': morans_i,
        'expected_i': expected_i,
        'lag': lag,
        'interpretation': 'clustered' if morans_i > expected_i else 'dispersed'
    }


if __name__ == "__main__":
    print("Spatial Analysis Module")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Synthetic binary mask with some structure
    mask = np.zeros((256, 256))
    # Add some patches
    mask[50:100, 50:100] = 1
    mask[150:200, 100:180] = 1
    mask[30:60, 180:220] = 1
    # Add some noise
    mask += (np.random.rand(256, 256) > 0.95).astype(float)
    mask = (mask > 0).astype(int)
    
    # Fractal dimension
    print("\nFractal Dimension Analysis:")
    fd_results = fractal_dimension_box_counting(mask)
    print(f"  Fractal Dimension: {fd_results['fractal_dimension']:.3f}")
    print(f"  R²: {fd_results['r_squared']:.3f}")
    
    # Lacunarity
    print("\nLacunarity Analysis:")
    lac_results = lacunarity_analysis(mask)
    print(f"  Mean Lacunarity: {lac_results['mean_lacunarity']:.3f}")
    print(f"  Lacunarity Slope: {lac_results['lacunarity_slope']:.3f}")
    
    # Landscape metrics
    print("\nLandscape Metrics:")
    lm_results = landscape_metrics(mask)
    print(f"  Number of Patches: {lm_results.get('num_patches', 'N/A')}")
    print(f"  Total Area: {lm_results['total_area_km2']:.3f} km²")
    print(f"  Coverage: {lm_results['coverage_percent']:.1f}%")
    
    # Markov chain example
    print("\nMarkov Chain Analysis:")
    # Simulate state sequence
    states = np.random.choice(5, size=(20, 100))  # 20 time steps, 100 pixels
    markov_results = markov_chain_analysis(states)
    print(f"  Total Transitions: {markov_results['n_transitions']}")
    print(f"  Stationary Distribution: {np.round(markov_results['stationary_distribution'], 3)}")
