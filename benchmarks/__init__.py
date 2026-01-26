"""
Benchmark models for comparison with the proposed CNN encoder-decoder.

This module provides implementations of traditional machine learning methods
(Random Forest, SVM, XGBoost) and deep learning architectures (VGG-16, ResNet-50,
U-Net, DeepLabV3+) used for comparative analysis.

Reference:
    Supplementary Material S1.8: Comparative Model Analysis
"""

from .traditional_ml import (
    build_random_forest,
    build_svm,
    build_xgboost,
    train_and_evaluate_model,
    cross_validate_model,
    run_all_traditional_ml_benchmarks,
    TRADITIONAL_ML_CONFIGS
)

from .deep_learning import (
    build_vgg16_classifier,
    build_resnet50_classifier,
    build_unet,
    build_deeplabv3plus,
    train_deep_model,
    evaluate_deep_model,
    run_deep_learning_benchmarks,
    DEEP_LEARNING_CONFIGS
)

__all__ = [
    # Traditional ML
    'build_random_forest',
    'build_svm',
    'build_xgboost',
    'train_and_evaluate_model',
    'cross_validate_model',
    'run_all_traditional_ml_benchmarks',
    'TRADITIONAL_ML_CONFIGS',
    # Deep Learning
    'build_vgg16_classifier',
    'build_resnet50_classifier',
    'build_unet',
    'build_deeplabv3plus',
    'train_deep_model',
    'evaluate_deep_model',
    'run_deep_learning_benchmarks',
    'DEEP_LEARNING_CONFIGS'
]
