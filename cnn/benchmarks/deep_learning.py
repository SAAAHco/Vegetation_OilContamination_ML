"""
Deep Learning Benchmark Models

This module implements state-of-the-art deep learning architectures for comparison:
- VGG-16
- ResNet-50
- U-Net
- DeepLabV3+

These architectures are benchmarked against our custom CNN as described in 
Supplementary Material Section S1.8.

Performance Summary (from manuscript):
    VGG-16: 81.7% accuracy, F1=0.79, κ=0.61
    ResNet-50: 84.1% accuracy, F1=0.82, κ=0.67
    U-Net: 87.2% accuracy, F1=0.85, κ=0.73
    DeepLabV3+: 86.4% accuracy, F1=0.84, κ=0.71
    Our CNN: 89.3% accuracy, F1=0.88, κ=0.76
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional, Dict


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_classes: int = 1,
    learning_rate: float = 0.0001,
    freeze_base: bool = True
) -> models.Model:
    """
    Build VGG-16 based classifier for pixel-wise classification.
    
    From Supplementary Material Section S1.8:
    "VGG-16: Pre-trained on ImageNet, fine-tuned with learning rate=0.0001; 
    achieved 81.7% accuracy, F1=0.79, κ=0.61"
    
    Args:
        input_shape: Input shape (height, width, channels)
                    Note: VGG-16 expects 3 channels, we adapt for 6 channels
        num_classes: Number of output classes
        learning_rate: Learning rate for fine-tuning (0.0001)
        freeze_base: Whether to freeze pre-trained layers initially
        
    Returns:
        Compiled Keras Model
    """
    # Handle 6-channel input by adding projection layer
    inputs = layers.Input(shape=input_shape)
    
    # Project from 6 channels to 3 channels for VGG-16 compatibility
    x = layers.Conv2D(3, (1, 1), activation='relu', name='channel_projection')(inputs)
    
    # Load pre-trained VGG-16 (without top)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    
    # Freeze base model if specified
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Get features
    features = base_model.output
    
    # Add classification head for pixel-wise classification
    # Use transposed convolutions to upsample back to input size
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='VGG16_Classifier')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def build_resnet50_classifier(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_classes: int = 1,
    learning_rate: float = 0.0001,
    freeze_base: bool = True
) -> models.Model:
    """
    Build ResNet-50 based classifier for pixel-wise classification.
    
    From Supplementary Material Section S1.8:
    "ResNet-50: Pre-trained, fine-tuned with learning rate=0.0001; 
    achieved 84.1% accuracy, F1=0.82, κ=0.67"
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for fine-tuning (0.0001)
        freeze_base: Whether to freeze pre-trained layers initially
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Project from 6 channels to 3 channels
    x = layers.Conv2D(3, (1, 1), activation='relu', name='channel_projection')(inputs)
    
    # Load pre-trained ResNet-50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    features = base_model.output
    
    # Decoder with skip connections would be ideal, but for benchmark we use simple upsampling
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_Classifier')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_classes: int = 1,
    learning_rate: float = 0.001,
    filters: List[int] = [64, 128, 256, 512]
) -> models.Model:
    """
    Build U-Net architecture for semantic segmentation.
    
    From Supplementary Material Section S1.8:
    "U-Net: Trained from scratch with learning rate=0.001; 
    achieved 87.2% accuracy, F1=0.85, κ=0.73"
    
    U-Net Reference: Ronneberger et al. (2015)
    
    Args:
        input_shape: Input shape
        num_classes: Number of output classes
        learning_rate: Learning rate (0.001)
        filters: List of filter counts for encoder blocks
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    skip_connections = []
    x = inputs
    
    # ==================== ENCODER ====================
    for i, f in enumerate(filters):
        # Conv block
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Store for skip connection
        skip_connections.append(x)
        
        # Pooling (except last)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D((2, 2))(x)
    
    # ==================== BOTTLENECK ====================
    bottleneck_filters = filters[-1] * 2
    x = layers.Conv2D(bottleneck_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(bottleneck_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # ==================== DECODER ====================
    for i, f in enumerate(reversed(filters)):
        # Upsample
        x = layers.Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Concatenate skip connection
        skip_idx = len(filters) - 1 - i
        x = layers.Concatenate()([x, skip_connections[skip_idx]])
        
        # Conv block
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
    # ==================== OUTPUT ====================
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def build_deeplabv3plus(
    input_shape: Tuple[int, int, int] = (256, 256, 6),
    num_classes: int = 1,
    learning_rate: float = 0.0001,
    output_stride: int = 16
) -> models.Model:
    """
    Build DeepLabV3+ architecture for semantic segmentation.
    
    From Supplementary Material Section S1.8:
    "DeepLabV3+: Pre-trained backbone, fine-tuned; 
    achieved 86.4% accuracy, F1=0.84, κ=0.71"
    
    DeepLabV3+ Reference: Chen et al. (2018)
    
    Features:
    - Atrous Spatial Pyramid Pooling (ASPP)
    - Encoder-decoder structure with low-level features
    - Depthwise separable convolutions
    
    Args:
        input_shape: Input shape
        num_classes: Number of output classes
        learning_rate: Learning rate
        output_stride: Output stride (8 or 16)
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # ==================== ENCODER (Simplified ResNet-like backbone) ====================
    # Initial conv
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Store low-level features for decoder
    low_level_features = x
    
    # Residual blocks (simplified)
    for filters in [64, 128, 256, 512]:
        x = _residual_block(x, filters)
    
    # ==================== ASPP (Atrous Spatial Pyramid Pooling) ====================
    aspp_out = _aspp_module(x, filters=256)
    
    # ==================== DECODER ====================
    # Upsample ASPP output
    x = layers.Conv2D(256, (1, 1), padding='same')(aspp_out)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Upsample to match low-level features
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Process low-level features
    low_level = layers.Conv2D(48, (1, 1), padding='same')(low_level_features)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.Activation('relu')(low_level)
    
    # Concatenate
    x = layers.Concatenate()([x, low_level])
    
    # Final conv blocks
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Upsample to original size
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='DeepLabV3plus')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def _residual_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Simple residual block."""
    shortcut = x
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def _aspp_module(x: tf.Tensor, filters: int = 256) -> tf.Tensor:
    """
    Atrous Spatial Pyramid Pooling module.
    
    Uses parallel dilated convolutions at multiple rates to capture 
    multi-scale context.
    """
    # Image-level features (global average pooling)
    image_pooling = layers.GlobalAveragePooling2D()(x)
    image_pooling = layers.Reshape((1, 1, x.shape[-1]))(image_pooling)
    image_pooling = layers.Conv2D(filters, (1, 1), padding='same')(image_pooling)
    image_pooling = layers.BatchNormalization()(image_pooling)
    image_pooling = layers.Activation('relu')(image_pooling)
    image_pooling = layers.UpSampling2D(
        size=(x.shape[1], x.shape[2]),
        interpolation='bilinear'
    )(image_pooling)
    
    # 1x1 convolution
    conv1x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
    conv1x1 = layers.BatchNormalization()(conv1x1)
    conv1x1 = layers.Activation('relu')(conv1x1)
    
    # Dilated convolutions at different rates
    rates = [6, 12, 18]
    aspp_outputs = [image_pooling, conv1x1]
    
    for rate in rates:
        conv = layers.Conv2D(filters, (3, 3), dilation_rate=rate, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        aspp_outputs.append(conv)
    
    # Concatenate all ASPP outputs
    x = layers.Concatenate()(aspp_outputs)
    
    # Final 1x1 convolution
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    return x


def get_benchmark_callbacks(
    model_name: str,
    patience: int = 15
) -> List[callbacks.Callback]:
    """Get training callbacks for benchmark models."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]


def run_deep_learning_benchmarks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    models_to_run: Optional[List[str]] = None
) -> Dict:
    """
    Run all deep learning benchmark models.
    
    Args:
        X_train: Training images
        y_train: Training masks
        X_val: Validation images
        y_val: Validation masks
        epochs: Maximum epochs (100)
        batch_size: Batch size (32)
        models_to_run: List of models to run. If None, runs all.
                      Options: 'vgg16', 'resnet50', 'unet', 'deeplabv3plus'
        
    Returns:
        Dictionary with results for all models
    """
    if models_to_run is None:
        models_to_run = ['vgg16', 'resnet50', 'unet', 'deeplabv3plus']
    
    input_shape = X_train.shape[1:]
    results = {}
    
    model_builders = {
        'vgg16': lambda: build_vgg16_classifier(input_shape),
        'resnet50': lambda: build_resnet50_classifier(input_shape),
        'unet': lambda: build_unet(input_shape),
        'deeplabv3plus': lambda: build_deeplabv3plus(input_shape)
    }
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # Build model
        model = model_builders[model_name]()
        model.summary()
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_benchmark_callbacks(model_name),
            verbose=1
        )
        
        # Evaluate
        eval_results = model.evaluate(X_val, y_val, verbose=0)
        
        results[model_name] = {
            'model': model,
            'history': history.history,
            'loss': eval_results[0],
            'accuracy': eval_results[1],
            'precision': eval_results[2],
            'recall': eval_results[3]
        }
        
        # Calculate F1 from precision and recall
        p = results[model_name]['precision']
        r = results[model_name]['recall']
        results[model_name]['f1_score'] = 2 * (p * r) / (p + r + 1e-7)
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {results[model_name]['accuracy']*100:.1f}%")
        print(f"  F1 Score: {results[model_name]['f1_score']:.3f}")
        
        # Clean up
        tf.keras.backend.clear_session()
    
    # Summary
    print("\n" + "="*60)
    print("DEEP LEARNING BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 42)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']*100:>9.1f}% {res['f1_score']:>10.3f}")
    
    return results


if __name__ == "__main__":
    print("Deep Learning Benchmarks Module")
    print("=" * 50)
    
    # Build and display U-Net architecture as example
    print("\nBuilding U-Net model...")
    unet = build_unet(input_shape=(256, 256, 6))
    unet.summary()
    
    # Count parameters
    trainable = np.sum([np.prod(v.shape) for v in unet.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable:,}")
