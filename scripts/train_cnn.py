#!/usr/bin/env python3
"""
End-to-end training script for CNN encoder-decoder model.

This script provides a complete training pipeline for the oil contamination
detection model as described in Section 2.3 of the manuscript.

Usage:
    python scripts/train_cnn.py --data_dir /path/to/data --output_dir /path/to/output
    python scripts/train_cnn.py --config config.yaml

Reference:
    Manuscript Section 2.3: Deep Learning Architecture
    Supplementary Material S1.4: Model Training Details
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig, TrainingConfig, DataConfig
from models.cnn_encoder_decoder import (
    build_cnn_encoder_decoder,
    build_classification_model,
    train_model_with_cross_validation
)
from preprocessing.data_loader import SatelliteDataLoader
from preprocessing.data_augmentation import DataAugmentor
from preprocessing.patch_extraction import PatchExtractor
from utils.metrics import calculate_all_metrics, calculate_confusion_matrix
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_prediction_samples
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN encoder-decoder for oil contamination detection'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory for saving outputs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Patch size in pixels (default: 256)'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_gpu(gpu_id: int):
    """Configure GPU settings."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            logger.info(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        else:
            logger.warning("No GPU available, using CPU")
    except Exception as e:
        logger.warning(f"GPU setup failed: {e}")


def load_data(data_dir: str, patch_size: int, augment: bool = False):
    """
    Load and prepare training data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing satellite images and labels
    patch_size : int
        Size of patches to extract
    augment : bool
        Whether to apply data augmentation
        
    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Initialize data loader
    loader = SatelliteDataLoader(
        data_dir=data_dir,
        bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    )
    
    # Load satellite images
    images, labels = loader.load_dataset()
    logger.info(f"Loaded {len(images)} images")
    
    # Extract patches
    extractor = PatchExtractor(
        patch_size=patch_size,
        overlap=0.25,  # 25% overlap for more training samples
        min_valid_ratio=0.8
    )
    
    X_patches, y_patches = extractor.extract_patches_from_dataset(images, labels)
    logger.info(f"Extracted {len(X_patches)} patches")
    
    # Split data: 60% train, 20% validation, 20% test
    n_samples = len(X_patches)
    indices = np.random.permutation(n_samples)
    
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train = X_patches[train_idx]
    y_train = y_patches[train_idx]
    X_val = X_patches[val_idx]
    y_val = y_patches[val_idx]
    X_test = X_patches[test_idx]
    y_test = y_patches[test_idx]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Apply data augmentation to training set
    if augment:
        logger.info("Applying data augmentation...")
        augmentor = DataAugmentor(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=0.15,
            noise_std=0.01
        )
        X_train, y_train = augmentor.augment_dataset(
            X_train, y_train,
            augmentation_factor=8
        )
        logger.info(f"After augmentation: {len(X_train)} training samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(args, X_train, y_train, X_val, y_val):
    """
    Train the CNN model.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
        
    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            ReduceLROnPlateau,
            TensorBoard,
            CSVLogger
        )
    except ImportError:
        raise ImportError("TensorFlow is required for training")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Build model
    logger.info("Building CNN encoder-decoder model...")
    input_shape = X_train.shape[1:]
    n_classes = len(np.unique(y_train))
    
    model = build_classification_model(
        input_shape=input_shape,
        n_classes=n_classes,
        filters=[32, 64, 128, 256, 512],
        dropout_rate=0.3
    )
    
    # Configure optimizer with cosine annealing
    # Reference: Manuscript Section 2.3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.epochs * len(X_train) // args.batch_size,
        alpha=0.01  # Minimum learning rate factor
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Model parameters: {model.count_params():,}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1
        ),
        CSVLogger(
            str(output_dir / 'training_history.csv')
        )
    ]
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model.load_weights(args.resume)
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(str(output_dir / 'model_final.keras'))
    logger.info(f"Model saved to {output_dir / 'model_final.keras'}")
    
    return model, history


def evaluate_model(model, X_test, y_test, output_dir: str):
    """
    Evaluate the trained model on test set.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    X_test, y_test : np.ndarray
        Test data
    output_dir : str
        Directory for saving evaluation results
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    output_dir = Path(output_dir)
    
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    
    # Calculate metrics
    # Reference: Supplementary Material S1.8
    metrics = calculate_all_metrics(y_test.flatten(), y_pred.flatten())
    
    logger.info("Test Set Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    
    # Save metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Calculate and save confusion matrix
    cm = calculate_confusion_matrix(y_test.flatten(), y_pred.flatten())
    np.save(output_dir / 'confusion_matrix.npy', cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        class_names=['Background', 'Contaminated', 'Recovering', 'Recovered'],
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    # Plot prediction samples
    plot_prediction_samples(
        X_test[:16], y_test[:16], y_pred[:16],
        save_path=str(output_dir / 'prediction_samples.png')
    )
    
    return metrics


def run_cross_validation(args, X, y):
    """
    Run k-fold cross-validation.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    X, y : np.ndarray
        Full dataset
        
    Returns
    -------
    dict
        Cross-validation results
    """
    logger.info(f"Running {args.n_folds}-fold cross-validation...")
    
    results = train_model_with_cross_validation(
        X, y,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info("Cross-validation Results:")
    logger.info(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    logger.info(f"  Mean F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    return results


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup
    logger.info("=" * 60)
    logger.info("CNN Encoder-Decoder Training Pipeline")
    logger.info("Oil Contamination Detection from Satellite Imagery")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    setup_gpu(args.gpu)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {args.output_dir}/config.json")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        args.data_dir,
        args.patch_size,
        augment=args.augment
    )
    
    # Train model
    model, history = train_model(args, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(
        history.history,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, args.output_dir)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Best test accuracy: {metrics['accuracy']:.4f}")
    logger.info("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()
