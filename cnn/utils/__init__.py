"""
Utility functions for oil-contaminated vegetation analysis.

This module provides evaluation metrics, visualization tools, and I/O utilities.
"""

from .metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_cohens_kappa,
    calculate_confusion_matrix,
    calculate_iou,
    calculate_dice_coefficient,
    calculate_all_metrics,
    mcnemar_test,
    bootstrap_confidence_interval,
    calculate_class_weights,
    per_class_metrics
)

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_prediction_samples,
    plot_roc_curves,
    plot_vegetation_indices_timeseries,
    plot_fft_spectrum,
    plot_cusum_change_detection,
    plot_recovery_phases,
    plot_spatial_gradient,
    plot_model_comparison,
    plot_segmentation_result,
    create_rgb_composite
)

__all__ = [
    # Metrics
    'calculate_accuracy',
    'calculate_precision',
    'calculate_recall',
    'calculate_f1_score',
    'calculate_cohens_kappa',
    'calculate_confusion_matrix',
    'calculate_iou',
    'calculate_dice_coefficient',
    'calculate_all_metrics',
    'mcnemar_test',
    'bootstrap_confidence_interval',
    'calculate_class_weights',
    'per_class_metrics',
    # Visualization
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_prediction_samples',
    'plot_roc_curves',
    'plot_vegetation_indices_timeseries',
    'plot_fft_spectrum',
    'plot_cusum_change_detection',
    'plot_recovery_phases',
    'plot_spatial_gradient',
    'plot_model_comparison',
    'plot_segmentation_result',
    'create_rgb_composite'
]
