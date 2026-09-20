"""
Visualization utilities for oil contamination detection analysis.

This module provides plotting functions for model evaluation, temporal analysis,
and spatial pattern visualization.

Reference:
    Manuscript Section 3: Results and Discussion
    Supplementary Figures S1-S10
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


def plot_training_history(history: Dict[str, List[float]], 
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot training and validation metrics over epochs.
    
    Parameters
    ----------
    history : dict
        Training history dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history.get('loss', []), label='Training', linewidth=2)
    axes[0].plot(history.get('val_loss', []), label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.get('accuracy', []), label='Training', linewidth=2)
    axes[1].plot(history.get('val_accuracy', []), label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, 
                          class_names: Optional[List[str]] = None,
                          normalize: bool = False,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6),
                          cmap: str = 'Blues') -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Reference:
        Supplementary Material S1.8, Figure S8
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes)
    class_names : list, optional
        Names for each class
    normalize : bool
        Whether to normalize by row (true labels)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name
    """
    import matplotlib.pyplot as plt
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_prediction_samples(images: np.ndarray, 
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            n_samples: int = 16,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 16)) -> None:
    """
    Plot sample predictions alongside ground truth.
    
    Parameters
    ----------
    images : np.ndarray
        Input images of shape (N, H, W, C)
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    n_samples : int
        Number of samples to display
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Display RGB composite (bands 4, 3, 2 if 6 bands)
        if images.shape[-1] >= 3:
            rgb = images[i, :, :, :3]
            # Normalize for display
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        else:
            rgb = images[i, :, :, 0]
        
        ax.imshow(rgb)
        
        # Title with true/predicted labels
        correct = y_true[i] == y_pred[i]
        color = 'green' if np.all(correct) else 'red'
        ax.set_title(f'True: {y_true[i].flatten()[0]}, Pred: {y_pred[i].flatten()[0]}',
                     color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels
    y_scores : dict
        Dictionary mapping model names to their predicted scores
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(y_scores)))
    
    for (name, scores), color in zip(y_scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_vegetation_indices_timeseries(dates: np.ndarray,
                                        indices: Dict[str, np.ndarray],
                                        save_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot vegetation indices time series.
    
    Reference:
        Manuscript Figure 4: Temporal Evolution of Vegetation Indices
    
    Parameters
    ----------
    dates : np.ndarray
        Array of dates
    indices : dict
        Dictionary mapping index names (ARVI, SAVI, HCI) to their time series
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(len(indices), 1, figsize=figsize, sharex=True)
    if len(indices) == 1:
        axes = [axes]
    
    colors = {'ARVI': '#2E86AB', 'SAVI': '#A23B72', 'HCI': '#F18F01'}
    
    for ax, (name, values) in zip(axes, indices.items()):
        color = colors.get(name, '#333333')
        ax.plot(dates, values, color=color, linewidth=1.5, label=name)
        ax.fill_between(dates, values, alpha=0.3, color=color)
        ax.set_ylabel(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.suptitle('Vegetation Index Time Series', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_fft_spectrum(frequencies: np.ndarray, power: np.ndarray,
                      significant_peaks: Optional[List[float]] = None,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 5)) -> None:
    """
    Plot FFT power spectrum.
    
    Reference:
        Supplementary Material S1.5, Equation 12
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values (cycles/year)
    power : np.ndarray
        Power spectral density
    significant_peaks : list, optional
        Frequencies of significant peaks to highlight
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(frequencies, power, 'b-', linewidth=1.5)
    ax.fill_between(frequencies, power, alpha=0.3)
    
    if significant_peaks:
        for peak_freq in significant_peaks:
            idx = np.argmin(np.abs(frequencies - peak_freq))
            ax.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.7)
            ax.annotate(f'{peak_freq:.2f} cycles/yr',
                        xy=(peak_freq, power[idx]),
                        xytext=(peak_freq + 0.5, power[idx] * 1.1),
                        fontsize=9, color='red')
    
    ax.set_xlabel('Frequency (cycles/year)', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.set_title('FFT Power Spectrum Analysis', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(frequencies))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_cusum_change_detection(dates: np.ndarray, values: np.ndarray,
                                 cusum: np.ndarray, change_points: List[int],
                                 threshold: float,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot CUSUM change detection results.
    
    Reference:
        Supplementary Material S1.5, Equation 13
    
    Parameters
    ----------
    dates : np.ndarray
        Date values
    values : np.ndarray
        Original time series values
    cusum : np.ndarray
        CUSUM statistic values
    change_points : list
        Indices of detected change points
    threshold : float
        Detection threshold (h parameter)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Original time series
    axes[0].plot(dates, values, 'b-', linewidth=1.5)
    for cp in change_points:
        axes[0].axvline(x=dates[cp], color='r', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Original Time Series', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # CUSUM statistic
    axes[1].plot(dates, cusum, 'g-', linewidth=1.5, label='CUSUM')
    axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (h={threshold:.2f})')
    for cp in change_points:
        axes[1].axvline(x=dates[cp], color='r', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('CUSUM Statistic', fontsize=12)
    axes[1].set_title('CUSUM Change Detection', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_recovery_phases(dates: np.ndarray, area: np.ndarray,
                          phases: Dict[str, Tuple[int, int]],
                          trends: Dict[str, float],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot recovery phases with linear trends.
    
    Reference:
        Supplementary Material S1.5: Recovery Phase Analysis
    
    Parameters
    ----------
    dates : np.ndarray
        Date values
    area : np.ndarray
        Contaminated area values (km²)
    phases : dict
        Dictionary mapping phase names to (start_idx, end_idx) tuples
    trends : dict
        Dictionary mapping phase names to linear trend slopes (km²/year)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot overall time series
    ax.plot(dates, area, 'ko-', markersize=4, linewidth=1, alpha=0.5, label='Data')
    
    colors = ['#E63946', '#457B9D', '#2A9D8F']
    
    for (phase_name, (start, end)), color in zip(phases.items(), colors):
        phase_dates = dates[start:end+1]
        phase_area = area[start:end+1]
        
        # Fit and plot trend line
        x_numeric = np.arange(len(phase_area))
        coeffs = np.polyfit(x_numeric, phase_area, 1)
        trend_line = np.polyval(coeffs, x_numeric)
        
        ax.plot(phase_dates, trend_line, color=color, linewidth=2.5,
                label=f'{phase_name}: {trends[phase_name]:.3f} km²/yr')
        ax.fill_between(phase_dates, phase_area, alpha=0.2, color=color)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Contaminated Area (km²)', fontsize=12)
    ax.set_title('Recovery Phase Analysis', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_spatial_gradient(distances: np.ndarray, values: np.ndarray,
                           gradient_type: str = 'vegetation',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot spatial gradient from contamination source.
    
    Reference:
        Supplementary Material S1.7.6: Gradient Analysis
    
    Parameters
    ----------
    distances : np.ndarray
        Distance from contamination source (meters)
    values : np.ndarray
        Vegetation index or other metric values
    gradient_type : str
        Type of gradient ('vegetation' or 'contamination')
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(distances, values, 'o-', markersize=6, linewidth=2, color='#2E86AB')
    
    # Add smoothed trend line
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(values, sigma=2)
    ax.plot(distances, smoothed, '--', linewidth=2, color='#F18F01', 
            label='Smoothed trend')
    
    ax.set_xlabel('Distance from Contamination Source (m)', fontsize=12)
    ax.set_ylabel(f'{gradient_type.title()} Index', fontsize=12)
    ax.set_title(f'Spatial Gradient Analysis: {gradient_type.title()}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Highlight optimal recovery zone (2000-4000m from manuscript)
    ax.axvspan(2000, 4000, alpha=0.2, color='green', label='Optimal Recovery Zone')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_model_comparison(models: List[str], metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot comparison of model performance metrics.
    
    Reference:
        Supplementary Material S1.8: Benchmark Comparison
    
    Parameters
    ----------
    models : list
        List of model names
    metrics : dict
        Dictionary mapping metric names to lists of values for each model
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    n_models = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_segmentation_result(image: np.ndarray, mask_true: np.ndarray,
                              mask_pred: np.ndarray,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot segmentation result comparison.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C)
    mask_true : np.ndarray
        Ground truth mask (H, W)
    mask_pred : np.ndarray
        Predicted mask (H, W)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Input image (RGB composite)
    if image.shape[-1] >= 3:
        rgb = image[:, :, :3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    else:
        rgb = image[:, :, 0]
    
    axes[0].imshow(rgb)
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(mask_true, cmap='viridis')
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(mask_pred, cmap='viridis')
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def create_rgb_composite(bands: Dict[str, np.ndarray],
                          r: str = 'B04', g: str = 'B03', b: str = 'B02',
                          percentile_clip: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Create RGB composite from multispectral bands.
    
    Parameters
    ----------
    bands : dict
        Dictionary mapping band names to arrays
    r, g, b : str
        Band names for red, green, blue channels
    percentile_clip : tuple
        Percentiles for contrast stretching
        
    Returns
    -------
    np.ndarray
        RGB composite image (H, W, 3) with values in [0, 1]
    """
    rgb = np.stack([bands[r], bands[g], bands[b]], axis=-1)
    
    # Percentile-based contrast stretching
    for i in range(3):
        channel = rgb[:, :, i]
        p_low, p_high = np.percentile(channel, percentile_clip)
        channel = np.clip(channel, p_low, p_high)
        channel = (channel - p_low) / (p_high - p_low + 1e-8)
        rgb[:, :, i] = channel
    
    return rgb


if __name__ == '__main__':
    # Example usage with synthetic data
    import matplotlib
    matplotlib.use('Agg')
    
    # Training history example
    history = {
        'loss': np.exp(-np.linspace(0, 2, 50)) + np.random.normal(0, 0.02, 50),
        'val_loss': np.exp(-np.linspace(0, 1.5, 50)) + np.random.normal(0, 0.03, 50),
        'accuracy': 1 - np.exp(-np.linspace(0, 2, 50)) * 0.5 + np.random.normal(0, 0.02, 50),
        'val_accuracy': 1 - np.exp(-np.linspace(0, 1.5, 50)) * 0.5 + np.random.normal(0, 0.03, 50)
    }
    plot_training_history(history, save_path='/tmp/training_history.png')
    print("Created training history plot")
    
    # Confusion matrix example
    cm = np.array([[85, 5, 3, 2],
                   [8, 78, 6, 3],
                   [4, 7, 82, 5],
                   [3, 2, 4, 88]])
    plot_confusion_matrix(cm, 
                          class_names=['Background', 'Contaminated', 'Recovering', 'Recovered'],
                          save_path='/tmp/confusion_matrix.png')
    print("Created confusion matrix plot")
    
    # Model comparison example
    models = ['SVM', 'RF', 'XGBoost', 'VGG-16', 'ResNet-50', 'U-Net', 'CNN (Ours)']
    metrics = {
        'Accuracy': [0.793, 0.820, 0.835, 0.817, 0.841, 0.872, 0.893],
        'F1 Score': [0.76, 0.79, 0.81, 0.79, 0.82, 0.85, 0.88],
        "Cohen's κ": [0.58, 0.62, 0.65, 0.61, 0.67, 0.73, 0.76]
    }
    plot_model_comparison(models, metrics, save_path='/tmp/model_comparison.png')
    print("Created model comparison plot")
    
    print("\nAll visualization examples completed successfully!")
